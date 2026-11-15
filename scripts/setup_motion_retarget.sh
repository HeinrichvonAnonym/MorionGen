#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MORIONGEN_DATA_ROOT:-/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen}"
CONDA="${CONDA_EXE:-/home/heinrich/anaconda3/bin/conda}"
STAGED="$DATA_ROOT/staged_third_party/python"
ENVS="$DATA_ROOT/envs"
PACKAGE_CACHE="$DATA_ROOT/package-cache"
ARIA2="$(dirname "$DATA_ROOT")/datasets/tools/aria2/root/usr/bin/aria2c"
ARIA2_LIB="$(dirname "$DATA_ROOT")/datasets/tools/aria2/root/usr/lib/x86_64-linux-gnu"
GVHMR="$STAGED/gvhmr"
WILOR="$STAGED/wilor"
SMPLX="/home/heinrich/RoboLab/python/benchmark/asset/human/smplx/models/smplx"
MANO="/home/heinrich/TactileLab/data/GRAB/mano_v1_2/models"
BASE_ENV="$ENVS/motiogen-smplsim"
SMPL="${MOTION_RETARGET_SMPL_DIR:-$DATA_ROOT/outputs/assets/closd-official/data/smpl}"

export CONDA_PKGS_DIRS="$PACKAGE_CACHE/conda"
export PIP_CACHE_DIR="$PACKAGE_CACHE/pip"
export TMPDIR="$PACKAGE_CACHE/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$STAGED" "$ENVS" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$ROOT/python/third_party"

clone_pinned() {
  local url="$1" path="$2" commit="$3"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  git -C "$path" fetch origin "$commit"
  git -C "$path" checkout --detach "$commit"
}

create_python_env() {
  local target="$1"
  if [[ -x "$BASE_ENV/bin/python" ]]; then
    "$BASE_ENV/bin/python" -m venv --system-site-packages "$target"
  else
    "$CONDA" create -p "$target" --override-channels -c https://repo.anaconda.com/pkgs/main python=3.10 pip -y
  fi
}

download_resume() {
  local url="$1" output="$2" attempt=1
  if [[ -x "$ARIA2" ]]; then
    LD_LIBRARY_PATH="$ARIA2_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$ARIA2" "$url" \
      --dir="$(dirname "$output")" --out="$(basename "$output")" \
      --continue=true --always-resume=true --split=8 --max-connection-per-server=8 \
      --min-split-size=1M --file-allocation=none --auto-file-renaming=false \
      --allow-overwrite=true --max-tries=20 --retry-wait=5 \
      --connect-timeout=60 --timeout=60
    return
  fi
  until curl -fL -C - --connect-timeout 60 --speed-time 120 --speed-limit 1024 \
      -o "$output" "$url"; do
    if [[ "$attempt" -ge 20 ]]; then
      echo "Download failed after $attempt resumable attempts: $url" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    sleep 3
  done
}

verify_size() {
  local path="$1" expected="$2" actual
  if [[ ! -f "$path" ]]; then
    echo "Downloaded asset is missing: $path" >&2
    return 1
  fi
  actual="$(stat -c '%s' "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Downloaded asset has unexpected size: $path ($actual, expected $expected bytes)" >&2
    return 1
  fi
}

verify_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | cut -d ' ' -f 1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Downloaded asset has unexpected SHA-256: $path ($actual, expected $expected)" >&2
    return 1
  fi
}

download_pypi_wheel() {
  local package="$1" version="$2" filename="$3" output url expected
  output="$PACKAGE_CACHE/wheels/$filename"
  read -r url expected < <(python3 - "$package" "$version" "$filename" <<'PY'
import json
import sys
import urllib.request

package, version, filename = sys.argv[1:]
with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json") as response:
    metadata = json.load(response)
matches = [item for item in metadata["urls"] if item["filename"] == filename]
if len(matches) != 1:
    raise SystemExit(f"Expected one PyPI wheel for {package}=={version}: {filename}")
print(matches[0]["url"], matches[0]["size"])
PY
  )
  if [[ ! -f "$output" ]] || [[ "$(stat -c '%s' "$output")" != "$expected" ]]; then
    download_resume "$url" "$output.partial"
    verify_size "$output.partial" "$expected"
    mv "$output.partial" "$output"
  fi
  verify_size "$output" "$expected"
  PYPI_WHEELS+=("$output")
}

clone_pinned https://github.com/zju3dv/GVHMR.git "$GVHMR" ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
# DPVO is an optional submodule and the official demo does not recommend it for
# fast inference. The pipeline deliberately uses GVHMR's default SimpleVO path.
clone_pinned https://github.com/rolpotamias/WiLoR.git "$WILOR" fcb911312a38fa8badd30d9656a167485d61b8f9
ln -sfn "$GVHMR" "$ROOT/python/third_party/gvhmr"
ln -sfn "$WILOR" "$ROOT/python/third_party/wilor"

if [[ ! -x "$ENVS/motiogen-wilor/bin/python" ]]; then
  create_python_env "$ENVS/motiogen-wilor"
fi
if ! "$ENVS/motiogen-wilor/bin/python" -c 'import torch, torchvision' >/dev/null 2>&1; then
  "$ENVS/motiogen-wilor/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi
# Pin the optional demo UI separately below.  Installing the unbounded
# `gradio` requirement first upgrades huggingface-hub beyond the versions used
# by this source revision and then immediately downgrades it again.
"$ENVS/motiogen-wilor/bin/python" -m pip install \
  -r <(sed '/^[[:space:]]*gradio[[:space:]]*$/d' "$WILOR/requirements.txt") scipy dill
# WiLoR's transitive UI stack may otherwise select releases newer than the
# Lightning/Transformers constraints used by this pinned source revision.
"$ENVS/motiogen-wilor/bin/python" -m pip install \
  'gradio==4.44.1' 'huggingface-hub==0.36.0' \
  'fsspec[http]==2024.12.0' 'packaging==24.2'
"$ENVS/motiogen-wilor/bin/python" -m pip uninstall -y hf-gradio >/dev/null 2>&1 || true
ln -sfn "$MANO/MANO_RIGHT.pkl" "$WILOR/mano_data/MANO_RIGHT.pkl"
if [[ ! -f "$WILOR/pretrained_models/detector.pt" ]]; then
  download_resume https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
    "$WILOR/pretrained_models/detector.pt.partial"
  verify_size "$WILOR/pretrained_models/detector.pt.partial" 53582271
  mv "$WILOR/pretrained_models/detector.pt.partial" "$WILOR/pretrained_models/detector.pt"
fi
verify_size "$WILOR/pretrained_models/detector.pt" 53582271
if [[ ! -f "$WILOR/pretrained_models/wilor_final.ckpt" ]]; then
  download_resume https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt \
    "$WILOR/pretrained_models/wilor_final.ckpt.partial"
  verify_size "$WILOR/pretrained_models/wilor_final.ckpt.partial" 2564989533
  mv "$WILOR/pretrained_models/wilor_final.ckpt.partial" "$WILOR/pretrained_models/wilor_final.ckpt"
fi
verify_size "$WILOR/pretrained_models/wilor_final.ckpt" 2564989533

if [[ ! -x "$ENVS/motiogen-gvhmr/bin/python" ]]; then
  "$CONDA" create -p "$ENVS/motiogen-gvhmr" --override-channels -c https://repo.anaconda.com/pkgs/main python=3.10 pip -y
fi
GV_TORCH_WHEEL="$PACKAGE_CACHE/wheels/torch-2.3.0+cu121-cp310-cp310-linux_x86_64.whl"
GV_TORCHVISION_WHEEL="$PACKAGE_CACHE/wheels/torchvision-0.18.0+cu121-cp310-cp310-linux_x86_64.whl"
mkdir -p "$(dirname "$GV_TORCH_WHEEL")"
if [[ ! -f "$GV_TORCH_WHEEL" ]] || [[ "$(stat -c '%s' "$GV_TORCH_WHEEL")" != 780956465 ]]; then
  download_resume \
    'https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl' \
    "$GV_TORCH_WHEEL.partial"
  verify_size "$GV_TORCH_WHEEL.partial" 780956465
  mv "$GV_TORCH_WHEEL.partial" "$GV_TORCH_WHEEL"
fi
if [[ ! -f "$GV_TORCHVISION_WHEEL" ]] || [[ "$(stat -c '%s' "$GV_TORCHVISION_WHEEL")" != 6999977 ]]; then
  download_resume \
    'https://download.pytorch.org/whl/cu121/torchvision-0.18.0%2Bcu121-cp310-cp310-linux_x86_64.whl' \
    "$GV_TORCHVISION_WHEEL.partial"
  verify_size "$GV_TORCHVISION_WHEEL.partial" 6999977
  mv "$GV_TORCHVISION_WHEEL.partial" "$GV_TORCHVISION_WHEEL"
fi
PYPI_WHEELS=()
# Torch 2.3's CUDA wheel declares the NVIDIA runtime as separate, very large
# PyPI wheels.  Cache them with the same resumable downloader used above;
# plain pip uses a single non-resumable connection for these ~2 GiB assets.
download_pypi_wheel nvidia-cublas-cu12 12.1.3.1 \
  nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-cuda-cupti-cu12 12.1.105 \
  nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-cuda-nvrtc-cu12 12.1.105 \
  nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-cuda-runtime-cu12 12.1.105 \
  nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-cudnn-cu12 8.9.2.26 \
  nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-cufft-cu12 11.0.2.54 \
  nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-curand-cu12 10.3.2.106 \
  nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-cusolver-cu12 11.4.5.107 \
  nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-cusparse-cu12 12.1.0.106 \
  nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-nccl-cu12 2.20.5 \
  nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl
download_pypi_wheel nvidia-nvjitlink-cu12 12.1.105 \
  nvidia_nvjitlink_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
download_pypi_wheel nvidia-nvtx-cu12 12.1.105 \
  nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
download_pypi_wheel triton 2.3.0 \
  triton-2.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
"$ENVS/motiogen-gvhmr/bin/python" -m pip install "${PYPI_WHEELS[@]}"
"$ENVS/motiogen-gvhmr/bin/python" -m pip install "$GV_TORCH_WHEEL" "$GV_TORCHVISION_WHEEL"
"$ENVS/motiogen-gvhmr/bin/python" -m pip install -r "$GVHMR/requirements.txt"
"$ENVS/motiogen-gvhmr/bin/python" -m pip install -e "$GVHMR"
mkdir -p "$GVHMR/inputs/checkpoints/body_models"
ln -sfn "$SMPLX" "$GVHMR/inputs/checkpoints/body_models/smplx"
ln -sfn "$SMPL" "$GVHMR/inputs/checkpoints/body_models/smpl"
GVHMR_ASSETS=(
  "$GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
  "$GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
  "$GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
  "$GVHMR/inputs/checkpoints/yolo/yolov8x.pt"
)
GVHMR_ASSETS_MISSING=0
for asset in "${GVHMR_ASSETS[@]}"; do
  [[ -f "$asset" ]] || GVHMR_ASSETS_MISSING=1
done
if [[ "$GVHMR_ASSETS_MISSING" -eq 1 ]]; then
  "$ENVS/motiogen-gvhmr/bin/python" -m pip install gdown
  # gdown can return zero even when individual files hit the Drive quota, so
  # the checked, immutable fallback below remains authoritative.
  "$ENVS/motiogen-gvhmr/bin/gdown" --folder \
    https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD \
    -O "$GVHMR/inputs/checkpoints" || true
fi

# The official 4DAnyone release uses the unmodified GVHMR inference stack and
# publishes its complete inference closure at this immutable revision.  Use it
# only when GVHMR's primary Google Drive is quota-limited, and authenticate
# every byte against the release metadata rather than trusting a moving URL.
GVHMR_HF_REVISION="4c80e87b805a5f8461cf339cdbe2fb4249e585aa"
download_gvhmr_fallback() {
  local relative="$1" output="$2" expected_size="$3" expected_sha="$4"
  if [[ ! -f "$output" ]]; then
    mkdir -p "$(dirname "$output")"
    download_resume \
      "https://huggingface.co/AntResearch/4DAnyone/resolve/$GVHMR_HF_REVISION/gvhmr/$relative" \
      "$output.partial"
    verify_size "$output.partial" "$expected_size"
    verify_sha256 "$output.partial" "$expected_sha"
    mv "$output.partial" "$output"
  fi
  verify_size "$output" "$expected_size"
  verify_sha256 "$output" "$expected_sha"
}
download_gvhmr_fallback gvhmr_siga24_release.ckpt \
  "$GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" \
  163508011 4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91
download_gvhmr_fallback 'epoch=10-step=25000.ckpt' \
  "$GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt" \
  2709494041 2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e
download_gvhmr_fallback vitpose-h-multi-coco.pth \
  "$GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" \
  2549075546 50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc
download_gvhmr_fallback yolov8x.pt \
  "$GVHMR/inputs/checkpoints/yolo/yolov8x.pt" \
  136867539 c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b
for asset in "${GVHMR_ASSETS[@]}"; do
  [[ -f "$asset" ]] || { echo "GVHMR asset is missing after setup: $asset" >&2; exit 1; }
done

"$ENVS/motiogen-wilor/bin/python" -m pip check
"$ENVS/motiogen-gvhmr/bin/python" -m pip check
(cd "$WILOR" && "$ENVS/motiogen-wilor/bin/python" -c \
  'from wilor.models import WiLoR; print("WiLoR import OK")')
(cd "$GVHMR" && "$ENVS/motiogen-gvhmr/bin/python" -c \
  'from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL; print("GVHMR import OK")')
"$ENVS/motiogen-wilor/bin/python" -m pip freeze > "$ROOT/config/environments/locks/motion_retarget_wilor.txt"
"$ENVS/motiogen-gvhmr/bin/python" -m pip freeze > "$ROOT/config/environments/locks/motion_retarget_gvhmr.txt"

echo "WiLoR and GVHMR runtimes are ready under $ENVS."
