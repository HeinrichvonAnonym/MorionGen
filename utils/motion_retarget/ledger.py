from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .jsonutil import json_safe


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                dataset TEXT NOT NULL, sequence_id TEXT NOT NULL, profile TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                config_hash TEXT NOT NULL, output_path TEXT, metrics_json TEXT,
                error TEXT, updated_at REAL NOT NULL,
                PRIMARY KEY(dataset, sequence_id, profile)
            )"""
        )
        existing = {row[1] for row in self.connection.execute("PRAGMA table_info(tasks)")}
        for name in ("input_hash", "model_hash"):
            if name not in existing:
                self.connection.execute(f"ALTER TABLE tasks ADD COLUMN {name} TEXT")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS artifacts (
                path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL, updated_at REAL NOT NULL
            )"""
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def upsert(
        self, dataset: str, sequence_id: str, profile: str, status: str,
        config_hash: str, output_path: str | None = None,
        metrics: dict[str, Any] | None = None, error: str | None = None,
        increment_attempts: bool = False, input_hash: str | None = None,
        model_hash: str | None = None,
    ) -> None:
        attempts = 1 if increment_attempts else 0
        self.connection.execute(
            """INSERT INTO tasks(dataset,sequence_id,profile,status,attempts,config_hash,
                   output_path,metrics_json,error,updated_at,input_hash,model_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(dataset,sequence_id,profile) DO UPDATE SET
                   status=excluded.status,
                   attempts=tasks.attempts+excluded.attempts,
                   config_hash=excluded.config_hash, output_path=excluded.output_path,
                   metrics_json=excluded.metrics_json,error=excluded.error,updated_at=excluded.updated_at,
                   input_hash=COALESCE(excluded.input_hash,tasks.input_hash),
                   model_hash=COALESCE(excluded.model_hash,tasks.model_hash)""",
            (dataset, sequence_id, profile, status, attempts, config_hash, output_path,
             json.dumps(json_safe(metrics), sort_keys=True, allow_nan=False) if metrics is not None else None,
             error, time.time(), input_hash, model_hash),
        )
        self.connection.commit()

    def completed(
        self, dataset: str, sequence_id: str, profile: str, config_hash: str,
        input_hash: str | None = None, model_hash: str | None = None,
    ) -> bool:
        row = self.connection.execute(
            "SELECT status,config_hash,output_path,input_hash,model_hash FROM tasks "
            "WHERE dataset=? AND sequence_id=? AND profile=?",
            (dataset, sequence_id, profile),
        ).fetchone()
        return bool(
            row and row[0] in {"extracted", "complete"} and row[1] == config_hash and
            row[2] and Path(row[2]).is_file() and
            (input_hash is None or row[3] == input_hash) and
            (model_hash is None or row[4] == model_hash)
        )

    def rows(self) -> Iterable[dict[str, Any]]:
        columns = ("dataset", "sequence_id", "profile", "status", "attempts", "config_hash",
                   "output_path", "metrics_json", "error", "updated_at", "input_hash", "model_hash")
        query = "SELECT " + ",".join(columns) + " FROM tasks ORDER BY dataset,sequence_id,profile"
        for row in self.connection.execute(query):
            item = dict(zip(columns, row))
            item["metrics"] = json.loads(item.pop("metrics_json")) if item["metrics_json"] else None
            yield item

    def metrics(self, dataset: str, sequence_id: str, profile: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT metrics_json FROM tasks WHERE dataset=? AND sequence_id=? AND profile=?",
            (dataset, sequence_id, profile),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def sha256(self, path: Path) -> str:
        import hashlib
        stat = path.stat()
        row = self.connection.execute(
            "SELECT size,mtime_ns,sha256 FROM artifacts WHERE path=?", (str(path.resolve()),)
        ).fetchone()
        if row and row[0] == stat.st_size and row[1] == stat.st_mtime_ns:
            return str(row[2])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 << 20), b""):
                digest.update(block)
        value = digest.hexdigest()
        self.connection.execute(
            """INSERT INTO artifacts(path,size,mtime_ns,sha256,updated_at) VALUES(?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,
               sha256=excluded.sha256,updated_at=excluded.updated_at""",
            (str(path.resolve()), stat.st_size, stat.st_mtime_ns, value, time.time()),
        )
        self.connection.commit()
        return value
