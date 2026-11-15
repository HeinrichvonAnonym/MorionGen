#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.motion_retarget.adapters import inventory
from utils.motion_retarget.config import load_config
from utils.motion_retarget.ledger import Ledger
from utils.motion_retarget.gpu import initialize_gpu_worker, parse_gpu_ids
from utils.motion_retarget.hiphi_stream import (
    cleanup_stale_hiphi_stages, group_hiphi_by_archive, stage_hiphi_archive,
    stage_hiphi_object_archive,
)
from utils.motion_retarget.object_pipeline import extract_object_reference
from utils.motion_retarget.objects import load_selection_manifest, select_references
from utils.motion_retarget.profiles import profiles_for_reference, validate_profile_requests
from utils.motion_retarget.pipeline import (
    extract_reference, input_dependency_hash, input_dependency_paths,
    inventory_payload, model_dependency_hash,
    output_path, prewarm_reference, run_config_hash, write_manifest,
)
from utils.motion_retarget.validation import validate_motion
from utils.motion_retarget.storage import write_json_atomic


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Native-first AnDy/COMFI/HiPHI to SMPL-X/MANO pipeline")
    result.add_argument("--config", type=Path, default=ROOT / "config/motion_retarget.yaml")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("inventory", "preflight", "prewarm", "extract", "validate", "render", "render-previews"):
        item = sub.add_parser(name)
        item.add_argument("--dataset", action="append", choices=("andy", "comfi", "hiphi"))
        item.add_argument("--sequence", action="append", help="Exact sequence id; repeatable")
        item.add_argument("--profile", action="append")
        item.add_argument("--allow-incomplete-hiphi", action="store_true")
        item.add_argument("--limit", type=int)
        item.add_argument("--only-status", action="append")
        item.add_argument("--input-mode", choices=("auto", "qualisys", "xsens_bvh", "mocap_aligned",
                                                    "mocap_raw", "bvh", "gvhmr_video"))
        item.add_argument("--with-objects", action="store_true")
        item.add_argument("--selection-manifest", type=Path)
        if name in {"prewarm", "extract"}:
            item.add_argument("--workers", type=int)
            item.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids")
        if name == "extract":
            item.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
            item.add_argument("--retries", type=int)
            item.add_argument(
                "--pilot-contact-windows", action="store_true",
                help="Run only acquisition/stable/release contact windows; use a test output root",
            )
        if name in {"validate", "render", "render-previews"}:
            item.add_argument("--mujoco", action=argparse.BooleanOptionalAction, default=True)
            item.add_argument("--workers", type=int, default=1)
    summary = sub.add_parser("summarize")
    summary.add_argument("--with-objects", action="store_true")
    summary.add_argument("--selection-manifest", type=Path)
    return result


def _object_refs(config, manifest_path: Path):
    manifest = load_selection_manifest(manifest_path)
    wanted = {str(item["sequence_id"]) for item in manifest["selected"]}
    refs = inventory(config, "hiphi", sequence_ids=wanted)
    return select_references(refs, manifest), manifest


def object_main(args, config) -> int:
    manifest_path = args.selection_manifest
    if manifest_path is None:
        configured = Path(str(config.raw["object_pilot"]["selection_manifest"]))
        manifest_path = configured if configured.is_absolute() else ROOT / configured
    refs, manifest = _object_refs(config, manifest_path.resolve())
    if getattr(args, "sequence", None):
        wanted = set(args.sequence)
        refs = [reference for reference in refs if reference.sequence_id in wanted]
        missing = wanted - {reference.sequence_id for reference in refs}
        if missing:
            raise SystemExit(f"Object pilot sequences not selected: {sorted(missing)}")
    if getattr(args, "limit", None) is not None:
        refs = refs[:args.limit]
    config.output_root.mkdir(parents=True, exist_ok=True)
    selected_payload = dict(manifest)
    selected_payload["resolved"] = [
        {"sequence_id": reference.sequence_id, "source": str(reference.source),
         "member": reference.member, "metadata": dict(reference.metadata)}
        for reference in refs
    ]
    write_json_atomic(config.output_root / "selection_manifest.json", selected_payload)
    inventory_report = {
        "schema_version": 4, "with_objects": True, "config_hash": config.digest,
        "expected_outputs": len(refs), "sequences": selected_payload["resolved"],
    }
    write_json_atomic(config.output_root / "inventory.json", inventory_report)
    if args.command == "inventory":
        print(json.dumps(inventory_report, indent=2))
        return 0
    ledger = Ledger(config.output_root / "ledger.sqlite3")
    failures = 0
    try:
        if args.command == "summarize":
            rows = list(ledger.rows())
            selected_keys = {
                ("hiphi", reference.sequence_id, "source_native")
                for reference in refs
            }
            selected_rows = [
                row for row in rows
                if (row["dataset"], row["sequence_id"], row["profile"]) in selected_keys
            ]
            status: dict[str, int] = {}
            for row in selected_rows:
                status[row["status"]] = status.get(row["status"], 0) + 1
            payload = {
                "with_objects": True, "expected": len(refs), "tasks": len(selected_rows),
                "status": status, "complete": status.get("complete", 0),
                "validation_failed": status.get("validation_failed", 0),
                "ledger_historical_unselected_tasks": len(rows) - len(selected_rows),
                "full_human_only_service_restored": False,
                "manifest": str(write_manifest(config, ledger)),
            }
            write_json_atomic(config.output_root / "summary.json", payload)
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "extract":
            if args.pilot_contact_windows and not config.output_root.is_relative_to(
                ROOT / "develop_logs/test_runs"
            ):
                raise SystemExit(
                    "--pilot-contact-windows requires MOTION_RETARGET_OUTPUT_ROOT under "
                    "develop_logs/test_runs so a partial solve cannot replace a deliverable"
                )
            workers = args.workers if args.workers is not None else int(config.raw["batch"]["workers"])
            retries = args.retries if args.retries is not None else int(config.raw["batch"]["retries"])
            gpu_ids = parse_gpu_ids(args.gpus) if args.gpus is not None else parse_gpu_ids(
                config.raw["batch"].get("gpus", [])
            )
            cleanup_stale_hiphi_stages(config)
            # Keep every selected shard's small staged subset alive while the
            # process pool runs.  The old implementation consumed each shard
            # synchronously and silently ignored --workers/--gpus.
            with ExitStack() as stages:
                staged_refs = []
                for archive_group in group_hiphi_by_archive(refs):
                    ledger.sha256(archive_group[0].source)
                    staged_refs.extend(stages.enter_context(
                        stage_hiphi_object_archive(config, archive_group)
                    ))
                if workers <= 1:
                    if gpu_ids:
                        os.environ["MOTION_RETARGET_GPU_INDEX"] = str(gpu_ids[0])
                    results = [
                        extract_object_reference(
                            config.path, reference, args.resume, retries,
                            args.pilot_contact_windows,
                        )
                        for reference in staged_refs
                    ]
                else:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=initialize_gpu_worker,
                        initargs=(gpu_ids,),
                    ) as executor:
                        futures = {
                            executor.submit(
                                extract_object_reference, config.path, reference,
                                args.resume, retries, args.pilot_contact_windows,
                            ): reference
                            for reference in staged_refs
                        }
                        results = []
                        for future in concurrent.futures.as_completed(futures):
                            reference = futures[future]
                            try:
                                results.append(future.result())
                            except Exception as exc:
                                results.append({
                                    "status": "failed", "sequence_id": reference.sequence_id,
                                    "error": f"worker failure: {exc}",
                                })
                for result in results:
                    print(json.dumps(
                        result, sort_keys=True,
                        default=lambda value: (
                            value.tolist() if hasattr(value, "tolist") else str(value)
                        ),
                    ), flush=True)
                    failures += result["status"] == "failed"
        elif args.command in {"validate", "render", "render-previews"}:
            validation_stack = ExitStack()
            validation_refs = refs
            if args.command == "validate":
                validation_refs = []
                for archive_group in group_hiphi_by_archive(refs):
                    validation_refs.extend(validation_stack.enter_context(
                        stage_hiphi_object_archive(config, archive_group)
                    ))
            for reference in validation_refs:
                profile = profiles_for_reference(config, reference, ["source_native"])[0]
                path = output_path(config, reference, profile)
                if not path.is_file():
                    print(json.dumps({"status": "missing", "output": str(path)}), flush=True)
                    failures += 1
                    continue
                render = path.with_suffix(".objects.mp4") if args.command in {"render", "render-previews"} else None
                try:
                    report = validate_motion(
                        config, path, args.mujoco, render, config.digest,
                        source_reference=(reference if args.command == "validate" else None),
                    )
                    archive_report = report.get("archive", {})
                    pilot_only = bool(
                        archive_report.get("quality", {}).get(
                            "constrained_object_solver", {}
                        ).get("pilot_windows", False)
                    )
                    status = (
                        "pilot_windows_passed" if report["passed"] and pilot_only
                        else (
                            "awaiting_visual_review" if report["passed"] and
                            int(archive_report.get("schema_version", 0)) >= 5
                            else ("complete" if report["passed"] else "validation_failed")
                        )
                    )
                    metrics = dict(ledger.metrics("hiphi", reference.sequence_id, profile.name) or {})
                    metrics["validation"] = report
                    ledger.upsert("hiphi", reference.sequence_id, profile.name, status, config.digest,
                                  str(path), metrics)
                    print(json.dumps({"status": status, "output": str(path), "report": report}, sort_keys=True), flush=True)
                    failures += status == "validation_failed"
                except Exception as exc:
                    ledger.upsert("hiphi", reference.sequence_id, profile.name, "validation_failed",
                                  config.digest, str(path), error=str(exc))
                    print(json.dumps({"status": "validation_failed", "output": str(path), "error": str(exc)}), flush=True)
                    failures += 1
            validation_stack.close()
        else:
            raise SystemExit(f"--with-objects does not support {args.command}")
        write_manifest(config, ledger)
        return min(1, failures)
    finally:
        ledger.close()


def selected(args, config):
    datasets = args.dataset or ["andy", "comfi", "hiphi"]
    try:
        profile_requests = validate_profile_requests(config, args.profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    refs = []
    found: set[str] = set()
    for dataset in datasets:
        wanted = (set(args.sequence) - found) if args.sequence else None
        if wanted is not None and not wanted:
            break
        dataset_refs = inventory(
            config, dataset, allow_incomplete=args.allow_incomplete_hiphi,
            sequence_ids=wanted,
        )
        refs.extend(dataset_refs)
        found.update(reference.sequence_id for reference in dataset_refs)
    if args.sequence:
        wanted = set(args.sequence)
        refs = [item for item in refs if item.sequence_id in wanted]
        missing = wanted - {item.sequence_id for item in refs}
        if missing:
            raise SystemExit(f"Sequences not found: {sorted(missing)}")
    if args.limit is not None:
        refs = refs[: args.limit]
    if args.only_status:
        ledger = Ledger(config.output_root / "ledger.sqlite3")
        try:
            wanted = {
                (row["dataset"], row["sequence_id"], row["profile"])
                for row in ledger.rows() if row["status"] in set(args.only_status)
            }
        finally:
            ledger.close()
        refs = [
            reference for reference in refs
            if any(
                (reference.dataset, reference.sequence_id, profile.name) in wanted
                for profile in profiles_for_reference(config, reference, profile_requests)
            )
        ]
    return refs, profile_requests


def preflight_payload(config, datasets, profile_requests, allow_incomplete):
    payload = inventory_payload(
        config, datasets, allow_incomplete=allow_incomplete,
        profile_requests=profile_requests,
    )
    frame_count = 0
    frame_count_basis = {}
    for dataset, details in payload["datasets"].items():
        if dataset == "hiphi":
            frames = sum(
                int(item["metadata"].get("frame_count") or 0)
                for item in details["sequences"]
            )
            basis = "release metadata exact frame_count"
        elif dataset == "andy":
            frames = int(config.raw["batch"].get("andy_preflight_median_frames", 8898)) * int(
                details["sequence_count"]
            )
            basis = "pilot median frames times sequence count"
        else:
            frames = 0
            basis = "not estimated"
        frame_count += frames
        frame_count_basis[dataset] = {"frames": frames, "basis": basis}
    bytes_per_frame = int(config.raw["batch"].get("estimated_compressed_bytes_per_frame", 3000))
    predicted_bytes = frame_count * bytes_per_frame
    disk = shutil.disk_usage(config.output_root)
    reserve = int(float(config.raw["quality"]["disk_reserve_gib"]) * 1024 ** 3)
    expected = {name: data["expected_outputs"] for name, data in payload["datasets"].items()}
    checks = {
        "comfi_excluded": "comfi" not in datasets,
        "andy_expected_195": "andy" not in expected or expected["andy"] == 195,
        "hiphi_expected_17626": "hiphi" not in expected or expected["hiphi"] == 17626,
        "space_after_estimate_above_reserve": disk.free - predicted_bytes >= reserve,
    }
    report = {
        "passed": all(checks.values()), "checks": checks,
        "output_root": str(config.output_root), "profile_requests": list(profile_requests),
        "expected_outputs": expected, "estimated_frames": frame_count,
        "frame_count_basis": frame_count_basis,
        "estimate_bytes_per_frame": bytes_per_frame,
        "estimated_output_gib": predicted_bytes / 1024 ** 3,
        "disk_free_gib": disk.free / 1024 ** 3,
        "estimated_free_after_output_gib": (disk.free - predicted_bytes) / 1024 ** 3,
        "disk_reserve_gib": reserve / 1024 ** 3,
        "inventory_path": str(config.output_root / "inventory.json"),
    }
    write_json_atomic(config.output_root / "preflight.json", report)
    return report


def stratified_preview_refs(config, references, profile_requests):
    counts = {"andy": 6, "hiphi": 12}
    selected_refs = []
    ledger = Ledger(config.output_root / "ledger.sqlite3")
    try:
        metrics_by_key = {
            (row["dataset"], row["sequence_id"], row["profile"]): row.get("metrics") or {}
            for row in ledger.rows()
        }
    finally:
        ledger.close()
    for dataset, count in counts.items():
        candidates = []
        for reference in references:
            if reference.dataset != dataset:
                continue
            profiles = profiles_for_reference(config, reference, profile_requests)
            if any(output_path(config, reference, profile).is_file() for profile in profiles):
                candidates.append(reference)
        picked = []
        genders = sorted({str(reference.metadata.get("gender", "")).lower() for reference in candidates})
        quota = max(1, count // max(1, len(genders)))
        for gender in genders:
            group = [
                reference for reference in candidates
                if str(reference.metadata.get("gender", "")).lower() == gender
            ]
            group.sort(key=lambda item: item.sequence_id)
            seeds = []
            if group:
                seeds.extend([
                    min(group, key=lambda item: float(item.metadata.get("height_m") or 0.0)),
                    max(group, key=lambda item: float(item.metadata.get("height_m") or 0.0)),
                ])
                def quality_value(reference):
                    metrics = metrics_by_key.get((dataset, reference.sequence_id, "source_native"), {})
                    if dataset == "hiphi":
                        native = metrics.get("native_hand_fk", {})
                        return float(native.get("median_error_m", float("inf")))
                    coverage = metrics.get("hand_coverage", {})
                    return min(float(coverage.get("left", 0.0)), float(coverage.get("right", 0.0)))
                finite_quality = [reference for reference in group if np.isfinite(quality_value(reference))]
                if finite_quality:
                    seeds.extend([
                        min(finite_quality, key=quality_value), max(finite_quality, key=quality_value),
                    ])
            for key in ("setup", "is_hoi", "object_categories"):
                values = {}
                for reference in group:
                    values.setdefault(str(reference.metadata.get(key, "")), reference)
                seeds.extend(values[value] for value in sorted(values))
            unique = {reference.sequence_id: reference for reference in seeds}
            for reference in group:
                unique.setdefault(reference.sequence_id, reference)
                if len(unique) >= quota:
                    break
            picked.extend(list(unique.values())[:quota])
        unique = {reference.sequence_id: reference for reference in picked}
        for reference in sorted(candidates, key=lambda item: item.sequence_id):
            unique.setdefault(reference.sequence_id, reference)
            if len(unique) >= count:
                break
        selected_refs.extend(list(unique.values())[:count])
    return selected_refs


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config)
    if getattr(args, "with_objects", False):
        return object_main(args, config)
    ledger = Ledger(config.output_root / "ledger.sqlite3")
    try:
        if args.command == "inventory":
            datasets = args.dataset or ["andy", "comfi", "hiphi"]
            requests = validate_profile_requests(config, args.profile)
            print(json.dumps(inventory_payload(
                config, datasets, args.allow_incomplete_hiphi, requests,
            ), indent=2))
            return 0
        if args.command == "preflight":
            datasets = args.dataset or ["andy", "hiphi"]
            requests = validate_profile_requests(config, args.profile or ["source_native"])
            report = preflight_payload(config, datasets, requests, args.allow_incomplete_hiphi)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
            return 0
        if args.command == "summarize":
            rows = list(ledger.rows())
            counts: dict[str, int] = {}
            datasets: dict[str, dict] = {}
            bytes_per_frame = []
            for row in rows:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                dataset = datasets.setdefault(row["dataset"], {
                    "tasks": 0, "status": {}, "output_bytes": 0,
                    "pilot_frames": [], "pilot_bytes_per_frame": [],
                })
                dataset["tasks"] += 1
                dataset["status"][row["status"]] = dataset["status"].get(row["status"], 0) + 1
                metrics = row.get("metrics") or {}
                output_bytes, frames = metrics.get("output_bytes"), metrics.get("frames")
                if output_bytes:
                    dataset["output_bytes"] += int(output_bytes)
                if output_bytes and frames:
                    ratio = float(output_bytes) / int(frames)
                    bytes_per_frame.append(ratio)
                    dataset["pilot_frames"].append(int(frames))
                    dataset["pilot_bytes_per_frame"].append(ratio)
            disk = shutil.disk_usage(config.output_root)
            inventory_path = config.output_root / "inventory.json"
            inventory_data = json.loads(inventory_path.read_text()) if inventory_path.is_file() else {}
            inventory_current = inventory_data.get("config_hash") == config.digest
            estimated_total = 0.0
            for name, dataset in datasets.items():
                frames_values = sorted(dataset.pop("pilot_frames"))
                ratio_values = sorted(dataset.pop("pilot_bytes_per_frame"))
                dataset["pilot_median_frames"] = (
                    frames_values[len(frames_values) // 2] if frames_values else None
                )
                dataset["pilot_compressed_bytes_per_frame_median"] = (
                    ratio_values[len(ratio_values) // 2] if ratio_values else None
                )
                inventory_dataset = inventory_data.get("datasets", {}).get(name, {}) if inventory_current else {}
                expected_outputs = inventory_dataset.get("expected_outputs")
                dataset["inventory_expected_outputs"] = expected_outputs
                if expected_outputs and frames_values and ratio_values:
                    estimate = float(expected_outputs) * dataset["pilot_median_frames"] * dataset[
                        "pilot_compressed_bytes_per_frame_median"
                    ]
                    dataset["estimated_full_output_bytes"] = int(estimate)
                    estimated_total += estimate
            payload = {
                "tasks": len(rows), "status": counts, "datasets": datasets,
                "pilot_compressed_bytes_per_frame_median": (
                    sorted(bytes_per_frame)[len(bytes_per_frame) // 2] if bytes_per_frame else None
                ),
                "disk_free_gib": disk.free / 1024 ** 3,
                "disk_reserve_gib": float(config.raw["quality"]["disk_reserve_gib"]),
                "estimated_full_output_gib": estimated_total / 1024 ** 3 if estimated_total else None,
                "estimated_free_after_full_output_gib": (
                    (disk.free - estimated_total) / 1024 ** 3 if estimated_total else None
                ),
                "space_estimate_basis": "per-dataset pilot median frames times median compressed bytes/frame",
                "inventory_hash_matches_config": inventory_current,
                "manifest": str(write_manifest(config, ledger)),
            }
            print(json.dumps(payload, indent=2))
            return 0
        refs, profile_requests = selected(args, config)
        if args.command == "render-previews":
            refs = stratified_preview_refs(config, refs, profile_requests)
            write_json_atomic(config.output_root / "preview_selection.json", {
                "sequences": [
                    {"dataset": reference.dataset, "sequence_id": reference.sequence_id,
                     "metadata": dict(reference.metadata)}
                    for reference in refs
                ],
            })
        failures = 0
        if args.command == "prewarm":
            workers = args.workers if args.workers is not None else int(config.raw["batch"]["workers"])
            gpu_ids = parse_gpu_ids(args.gpus) if args.gpus is not None else parse_gpu_ids(
                config.raw["batch"].get("gpus", [])
            )
            if workers <= 1:
                if gpu_ids:
                    os.environ["MOTION_RETARGET_GPU_INDEX"] = str(gpu_ids[0])
                results = [prewarm_reference(config.path, reference, args.input_mode) for reference in refs]
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers, initializer=initialize_gpu_worker, initargs=(gpu_ids,),
                ) as executor:
                    results = list(executor.map(
                        prewarm_reference, [config.path] * len(refs), refs,
                        [args.input_mode] * len(refs),
                    ))
            for result in results:
                print(json.dumps(result, sort_keys=True), flush=True)
                failures += result["status"] == "failed"
            return min(1, failures)
        if args.command == "extract":
            workers = args.workers if args.workers is not None else int(config.raw["batch"]["workers"])
            retries = args.retries if args.retries is not None else int(config.raw["batch"]["retries"])
            gpu_ids = parse_gpu_ids(args.gpus) if args.gpus is not None else parse_gpu_ids(
                config.raw["batch"].get("gpus", [])
            )
            def execute_references(batch):
                if workers <= 1:
                    if gpu_ids:
                        os.environ["MOTION_RETARGET_GPU_INDEX"] = str(gpu_ids[0])
                    return [extract_reference(config.path, reference, profile_requests, args.resume, retries,
                                              args.input_mode) for reference in batch]
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers, initializer=initialize_gpu_worker, initargs=(gpu_ids,),
                ) as executor:
                    futures = {
                        executor.submit(
                            extract_reference, config.path, reference, profile_requests,
                            args.resume, retries, args.input_mode,
                        ): reference
                        for reference in batch
                    }
                    groups = []
                    for future in concurrent.futures.as_completed(futures):
                        reference = futures[future]
                        try:
                            groups.append(future.result())
                        except Exception as exc:
                            error = f"worker_process_failure: {type(exc).__name__}: {exc}"
                            group = []
                            for profile in profiles_for_reference(config, reference, profile_requests):
                                ledger.upsert(
                                    reference.dataset, reference.sequence_id, profile.name,
                                    "failed", run_config_hash(config, reference, args.input_mode),
                                    error=error,
                                )
                                group.append({
                                    "status": "failed", "sequence_id": reference.sequence_id,
                                    "profile": profile.name, "error": error,
                                })
                            groups.append(group)
                    return groups

            native_refs = [reference for reference in refs if reference.dataset != "hiphi"]
            # Warm artifact hashes once in the parent.  Without this, several
            # workers can redundantly hash the same multi-gigabyte ZIP before
            # any of them has populated the shared artifact cache.
            for source in sorted({
                source
                for reference in native_refs
                for source in input_dependency_paths(config, reference, args.input_mode)
            }):
                ledger.sha256(source)
            for dataset in sorted({reference.dataset for reference in native_refs}):
                representative = next(reference for reference in native_refs if reference.dataset == dataset)
                for profile in profiles_for_reference(config, representative, profile_requests):
                    model_dependency_hash(config, ledger, representative, profile, args.input_mode)
            result_groups = execute_references(native_refs)
            hiphi_refs = [reference for reference in refs if reference.dataset == "hiphi"]
            if args.resume:
                retained = []
                for reference in hiphi_refs:
                    resolved_profiles = profiles_for_reference(config, reference, profile_requests)
                    input_hash = input_dependency_hash(
                        config, ledger, reference, args.input_mode,
                    )
                    model_hashes = {
                        profile.name: model_dependency_hash(
                            config, ledger, reference, profile, args.input_mode,
                        )
                        for profile in resolved_profiles
                    }
                    if any(not ledger.completed(
                        reference.dataset, reference.sequence_id, profile.name,
                        run_config_hash(config, reference, args.input_mode),
                        input_hash, model_hashes[profile.name],
                    ) for profile in resolved_profiles):
                        retained.append(reference)
                hiphi_refs = retained
            cleanup_stale_hiphi_stages(config)
            for archive_group in group_hiphi_by_archive(hiphi_refs):
                # Hash the immutable source shard before staging.  Staged BVHs
                # are temporary and must not become the resumability identity.
                ledger.sha256(archive_group[0].source)
                with stage_hiphi_archive(config, archive_group) as staged:
                    result_groups.extend(execute_references(staged))
            for group in result_groups:
                for result in group:
                    print(json.dumps(result, sort_keys=True), flush=True)
                    failures += result["status"] == "failed"
        else:
            jobs = [
                (reference, profile)
                for reference in refs
                for profile in profiles_for_reference(config, reference, profile_requests)
            ]

            def validate_job(job):
                reference, profile = job
                path = output_path(config, reference, profile)
                if not path.is_file():
                    return reference, profile, path, "missing", None, None
                render = path.with_suffix(".mp4") if args.command in {"render", "render-previews"} else None
                try:
                    report = validate_motion(
                        config, path, args.mujoco, render,
                        run_config_hash(config, reference, args.input_mode),
                    )
                    status = "complete" if report["passed"] else "validation_failed"
                    return reference, profile, path, status, report, None
                except Exception as exc:
                    return reference, profile, path, "validation_failed", None, str(exc)

            if args.workers <= 1:
                validation_results = map(validate_job, jobs)
            else:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
                validation_results = executor.map(validate_job, jobs)
            try:
                for reference, profile, path, status, report, error in validation_results:
                    if status != "missing":
                        metrics = dict(ledger.metrics(
                            reference.dataset, reference.sequence_id, profile.name,
                        ) or {})
                        metrics["validation"] = report if report is not None else {"error": error}
                        ledger.upsert(
                            reference.dataset, reference.sequence_id, profile.name, status,
                            run_config_hash(config, reference, args.input_mode),
                            str(path), metrics, error=error,
                        )
                    payload = {"status": status, "output": str(path)}
                    if report is not None:
                        payload["report"] = report
                    if error is not None:
                        payload["error"] = error
                    print(json.dumps(payload, sort_keys=True), flush=True)
                    failures += status != "complete"
            finally:
                if args.workers > 1:
                    executor.shutdown()
        write_manifest(config, ledger)
        return min(1, failures)
    finally:
        ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
