#!/usr/bin/env bash
set -euo pipefail
MOTIOGEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MOTIOGEN_ROOT"
MOTIOGEN_BASELINE="${1:?baseline name required}"
MOTIOGEN_DATA_ROOT="${MORIONGEN_DATA_ROOT:-$(/home/heinrich/anaconda3/bin/python -c 'import json,pathlib; print(pathlib.Path(json.load(open("config/baselines.json"))["data_root"]).expanduser().resolve())')}"
MOTIOGEN_ENV_ROOT="$MOTIOGEN_DATA_ROOT/envs"
MOTIOGEN_ENV="$MOTIOGEN_ENV_ROOT/motiogen-${MOTIOGEN_BASELINE}"
MOTIOGEN_ASSETS="$MOTIOGEN_DATA_ROOT/outputs/assets"
MOTIOGEN_PACKAGES="$MOTIOGEN_DATA_ROOT/package-cache"
MOTIOGEN_PYTHON=3.10
case "$MOTIOGEN_BASELINE" in
  motionlcm|omnicontrol|chois) ;;
  ardy|kimodo) MOTIOGEN_PYTHON=3.10 ;;
  *) exit 2 ;;
esac
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1 OMP_NUM_THREADS=1
export MORIONGEN_DATA_ROOT="$MOTIOGEN_DATA_ROOT"
export CONDA_PKGS_DIRS="$MOTIOGEN_PACKAGES/conda"
export PIP_CACHE_DIR="$MOTIOGEN_PACKAGES/pip"
export TMPDIR="$MOTIOGEN_PACKAGES/tmp"
mkdir -p "$MOTIOGEN_ENV_ROOT" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$MOTIOGEN_ASSETS/wheels"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${MOTIOGEN_PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_DEFAULT_TIMEOUT=30 PIP_RETRIES=2
if ! conda run -p "$MOTIOGEN_ENV" python -c 'import sys; print(sys.version)' >/dev/null 2>&1; then
  conda create -p "$MOTIOGEN_ENV" --override-channels -c https://repo.anaconda.com/pkgs/main "python=$MOTIOGEN_PYTHON" pip cmake ninja -y
fi
case "$MOTIOGEN_BASELINE" in
  motionlcm|omnicontrol|chois)
    MOTIOGEN_TORCH_WHEEL="$MOTIOGEN_ASSETS/wheels/torch-2.2.2+cu118-cp310-cp310-linux_x86_64.whl"
    if [[ -f "$MOTIOGEN_TORCH_WHEEL" ]]; then
      conda run --no-capture-output -p "$MOTIOGEN_ENV" python python/scripts/install_cached_torch.py "$MOTIOGEN_TORCH_WHEEL"
    else
      conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip download --no-deps --dest "$MOTIOGEN_ASSETS/wheels" --index-url https://download.pytorch.org/whl/cu118 torch==2.2.2+cu118
      MOTIOGEN_TORCH_WHEEL="$MOTIOGEN_ASSETS/wheels/torch-2.2.2+cu118-cp310-cp310-linux_x86_64.whl"
      conda run --no-capture-output -p "$MOTIOGEN_ENV" python python/scripts/install_cached_torch.py "$MOTIOGEN_TORCH_WHEEL"
    fi
    MOTIOGEN_REQUIREMENTS=config/environments/legacy.txt
    if [[ "$MOTIOGEN_BASELINE" == motionlcm ]]; then
      MOTIOGEN_REQUIREMENTS=config/environments/motionlcm.txt
    fi
    conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install -r "$MOTIOGEN_REQUIREMENTS" --find-links "$MOTIOGEN_ASSETS/wheels" --extra-index-url https://download.pytorch.org/whl/cu118
    ;;
  ardy|kimodo)
    conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install numpy==1.26.4 scipy==1.11.2 psutil==6.1.1 matplotlib==3.7.5
    /home/heinrich/anaconda3/bin/python python/scripts/stage_external_source.py "$MOTIOGEN_BASELINE"
    conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install -e "python/third_party/$MOTIOGEN_BASELINE"
    ;;
esac
if [[ "$MOTIOGEN_BASELINE" == omnicontrol || "$MOTIOGEN_BASELINE" == chois ]]; then
  # Chumpy's legacy setup imports pip while building, which fails in modern
  # isolated PEP-517 environments. It is still required to unpickle SMPL.
  conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install --no-build-isolation chumpy==0.70
  mkdir -p python/third_party
  MOTIOGEN_CLIP_SOURCE="$MOTIOGEN_DATA_ROOT/staged_third_party/python/CLIP"
  if [[ ! -d "$MOTIOGEN_CLIP_SOURCE/.git" ]]; then
    mkdir -p "$(dirname "$MOTIOGEN_CLIP_SOURCE")"
    git clone https://github.com/openai/CLIP.git "$MOTIOGEN_CLIP_SOURCE"
  fi
  if [[ ! -e python/third_party/CLIP ]]; then
    ln -s "$MOTIOGEN_CLIP_SOURCE" python/third_party/CLIP
  fi
  conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install -e python/third_party/CLIP --no-deps
fi
if [[ "$MOTIOGEN_BASELINE" == chois ]]; then
  /home/heinrich/anaconda3/bin/python python/scripts/stage_chois_dependencies.py
  conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install dotmap loguru PyOpenGL==3.1.0 fvcore iopath
  conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install -e python/third_party/human_body_prior -e python/third_party/bps_torch
  # Both CUDA extension setup.py files import the already-installed torch at
  # build time, so an isolated build environment cannot determine metadata.
  # CHOIS imports only pytorch3d.transforms (pure Python). The host nvcc 13.0
  # cannot build a binary extension for the pinned cu118 torch, so explicitly
  # omit pytorch3d._C instead of producing an ABI-incompatible shared object.
  PYTORCH3D_NO_EXTENSION=1 conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip install --no-build-isolation -e third_party/chamfer_distance -e third_party/pytorch3d
fi
conda run --no-capture-output -p "$MOTIOGEN_ENV" python -c 'import fcntl,torch; h=open("develop_logs/test_runs/.gpu.lock","a"); fcntl.flock(h,fcntl.LOCK_EX); print(torch.__version__); print(torch.cuda.get_device_name(0)); x=torch.ones(32,device="cuda"); print((x@x).item())'
mkdir -p config/environments/locks
conda run --no-capture-output -p "$MOTIOGEN_ENV" python -m pip freeze > "config/environments/locks/$MOTIOGEN_BASELINE.txt"
