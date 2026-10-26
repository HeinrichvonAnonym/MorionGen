#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DESTINATION="${2:?usage: restore_portable_bundle.sh PACKAGE_DIR DESTINATION}"

PACKAGE_DIR="$(cd "$PACKAGE_DIR" && pwd)"
mkdir -p "$DESTINATION"
DESTINATION="$(cd "$DESTINATION" && pwd)"

if ! compgen -G "$PACKAGE_DIR/moriongen-portable.tar.part-*" >/dev/null; then
  echo "No archive parts found below $PACKAGE_DIR" >&2
  exit 2
fi

(cd "$PACKAGE_DIR" && sha256sum -c SHA256SUMS)
cat "$PACKAGE_DIR"/moriongen-portable.tar.part-* | tar -xf - -C "$DESTINATION"

PORTABLE_ROOT="$DESTINATION/MorionGenPortable"
SMPL_LINK_DIR="$PORTABLE_ROOT/data/MorionGen/outputs/assets/chois/processed_data/smpl_all_models/smplx"
LLAMA_LINK_DIR="$PORTABLE_ROOT/data/MorionGen/outputs/assets/text/llm2vec/original_base"

for name in SMPLX_MALE.npz SMPLX_FEMALE.npz SMPLX_NEUTRAL.npz; do
  target="$PORTABLE_ROOT/licensed_assets/smplx/$name"
  relative="$(realpath -m --relative-to="$SMPL_LINK_DIR" "$target")"
  ln -sfn "$relative" "$SMPL_LINK_DIR/$name"
done

while IFS= read -r -d '' link; do
  target="$PORTABLE_ROOT/models/llama3-8b-instruct/$(basename "$link")"
  if [[ -e "$target" ]]; then
    relative="$(realpath -m --relative-to="$LLAMA_LINK_DIR" "$target")"
    ln -sfn "$relative" "$link"
  fi
done < <(find "$LLAMA_LINK_DIR" -maxdepth 1 -type l -print0)

cat > "$PORTABLE_ROOT/activate_portable.sh" <<EOF
#!/usr/bin/env bash
export MORIONGEN_DATA_ROOT="$PORTABLE_ROOT/data/MorionGen"
cd "$PORTABLE_ROOT/repo/MorionGen"
echo "MORIONGEN_DATA_ROOT=\$MORIONGEN_DATA_ROOT"
echo "Repository: \$PWD"
EOF
chmod +x "$PORTABLE_ROOT/activate_portable.sh"

echo "Restored successfully: $PORTABLE_ROOT"
echo "Run: source '$PORTABLE_ROOT/activate_portable.sh'"

