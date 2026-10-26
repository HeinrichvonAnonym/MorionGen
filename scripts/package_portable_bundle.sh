#!/usr/bin/env bash
set -euo pipefail

SOURCE_VOLUME="${MORIONGEN_SOURCE_VOLUME:-/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839}"
DATA_ROOT="${MORIONGEN_DATA_ROOT:-$SOURCE_VOLUME/MorionGen}"
USB_MOUNT="${MORIONGEN_USB_MOUNT:-/media/heinrich/KINGSTON}"
TARGET_NAME="${MORIONGEN_BUNDLE_NAME:-MorionGen_Portable_20261009}"
TARGET="$USB_MOUNT/$TARGET_NAME"
EXPECTED_USB_UUID="6024-EED1"
CHUNK_BYTES=3800000000

fail() { echo "ERROR: $*" >&2; exit 2; }

[[ -d "$SOURCE_VOLUME" ]] || fail "source volume is not mounted: $SOURCE_VOLUME"
[[ -d "$DATA_ROOT" ]] || fail "data root missing: $DATA_ROOT"
[[ -d "$USB_MOUNT" ]] || fail "USB mount missing: $USB_MOUNT"

actual_uuid="$(findmnt -no UUID --target "$USB_MOUNT")"
actual_fstype="$(findmnt -no FSTYPE --target "$USB_MOUNT")"
[[ "$actual_uuid" == "$EXPECTED_USB_UUID" ]] || fail "refuse unexpected USB UUID: $actual_uuid"
[[ "$actual_fstype" == "vfat" ]] || fail "refuse unexpected USB filesystem: $actual_fstype"
[[ ! -e "$TARGET" ]] || fail "target already exists; refusing overwrite: $TARGET"

required=(
  /home/heinrich/MorionGen
  "$DATA_ROOT/baselines"
  "$DATA_ROOT/staged_dependencies"
  "$DATA_ROOT/staged_third_party"
  "$DATA_ROOT/outputs/assets"
  "$DATA_ROOT/datasets"
  "$SOURCE_VOLUME/llama3-8b-instruct"
  /home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_MALE.npz
  /home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_FEMALE.npz
  /home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_NEUTRAL.npz
)
for path in "${required[@]}"; do [[ -e "$path" ]] || fail "required source missing: $path"; done

available_bytes="$(df -B1 --output=avail "$USB_MOUNT" | tail -1 | tr -d ' ')"
estimated_bytes=57000000000
(( available_bytes > estimated_bytes )) || fail "need at least $estimated_bytes bytes; available $available_bytes"

mkdir "$TARGET"
cp /home/heinrich/MorionGen/scripts/restore_portable_bundle.sh "$TARGET/restore_portable_bundle.sh"
chmod +x "$TARGET/restore_portable_bundle.sh"

cat > "$TARGET/README.txt" <<'EOF'
MorionGen portable source/assets bundle — 2026-10-09

Contents:
- MorionGen repository snapshot, including plans, configs, scripts and evidence notes.
- Five pinned official baseline Git repositories.
- Staged third-party source trees needed by CHOIS/ARDY/Kimodo.
- Runtime weights and extracted assets for MotionLCM, OmniControl, CHOIS, ARDY and Kimodo.
- Meta Llama 3 8B Instruct plus both LLM2Vec adapters.
- Prepared KIT-ML test data and the selected BONES-SEED stratified subset.
- The three licensed SMPL-X NPZ files referenced by CHOIS.

Deliberately excluded:
- Conda environments, package cache, failed downloads/checkouts/installs and generated benchmark outputs.
- Redundant downloaded .zip/.tar.gz/.rar archives when their extracted runtime files are included.
- Full 107 GB BONES-SEED source dataset and the 22,474-case Kimodo full benchmark extraction.

The USB filesystem is FAT32. The archive is split into 3.8 GB parts so source files larger
than 4 GB remain lossless inside the tar stream.

Restore on Linux:
  bash restore_portable_bundle.sh /path/to/this/directory /destination/with/enough/space

The restore script verifies SHA256SUMS, extracts the archive, repairs portable Llama/SMPL-X
symlinks, and creates MorionGenPortable/activate_portable.sh.
EOF

cat > "$TARGET/SCOPE.tsv" <<EOF
kind\tsource\tbundled_path
repo\t/home/heinrich/MorionGen\tMorionGenPortable/repo/MorionGen
baseline_repos\t$DATA_ROOT/baselines\tMorionGenPortable/data/MorionGen/baselines
third_party\t$DATA_ROOT/staged_dependencies\tMorionGenPortable/data/MorionGen/staged_dependencies
third_party\t$DATA_ROOT/staged_third_party\tMorionGenPortable/data/MorionGen/staged_third_party
assets_weights\t$DATA_ROOT/outputs/assets\tMorionGenPortable/data/MorionGen/outputs/assets
prepared_datasets\t$DATA_ROOT/datasets\tMorionGenPortable/data/MorionGen/datasets
llama3\t$SOURCE_VOLUME/llama3-8b-instruct\tMorionGenPortable/models/llama3-8b-instruct
licensed_smplx\t/home/heinrich/TactileLab/data/GRAB/smplx/models/smplx\tMorionGenPortable/licensed_assets/smplx
EOF

source_prefix="media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839"
tar \
  --exclude='*.zip' --exclude='*.tar.gz' --exclude='*.rar' --exclude='*.part' \
  --transform='s#^home/heinrich/MorionGen#MorionGenPortable/repo/MorionGen#' \
  --transform="s#^$source_prefix/MorionGen/#MorionGenPortable/data/MorionGen/#" \
  --transform="s#^$source_prefix/llama3-8b-instruct#MorionGenPortable/models/llama3-8b-instruct#" \
  --transform='s#^home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/#MorionGenPortable/licensed_assets/smplx/#' \
  -C / -cf - \
  home/heinrich/MorionGen \
  "$source_prefix/MorionGen/baselines" \
  "$source_prefix/MorionGen/staged_dependencies" \
  "$source_prefix/MorionGen/staged_third_party" \
  "$source_prefix/MorionGen/outputs/assets" \
  "$source_prefix/MorionGen/datasets" \
  "$source_prefix/llama3-8b-instruct" \
  home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_MALE.npz \
  home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_FEMALE.npz \
  home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_NEUTRAL.npz \
  | split --bytes="$CHUNK_BYTES" --numeric-suffixes=0 --suffix-length=3 - "$TARGET/moriongen-portable.tar.part-"

(cd "$TARGET" && sha256sum moriongen-portable.tar.part-* > SHA256SUMS)
(cd "$TARGET" && sha256sum -c SHA256SUMS)

part_count="$(find "$TARGET" -maxdepth 1 -name 'moriongen-portable.tar.part-*' -type f | wc -l)"
archive_bytes="$(find "$TARGET" -maxdepth 1 -name 'moriongen-portable.tar.part-*' -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f",s}')"
member_count="$(cat "$TARGET"/moriongen-portable.tar.part-* | tar -tf - | wc -l)"

cat > "$TARGET/VERIFICATION.txt" <<EOF
status=success
created_at=$(date --iso-8601=seconds)
usb_uuid=$actual_uuid
filesystem=$actual_fstype
archive_parts=$part_count
archive_bytes=$archive_bytes
tar_members=$member_count
chunk_bytes=$CHUNK_BYTES
sha256=verified
tar_listing=verified
EOF

sync "$TARGET"
echo "Portable bundle complete: $TARGET"
cat "$TARGET/VERIFICATION.txt"
