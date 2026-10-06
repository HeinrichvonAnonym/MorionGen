#!/usr/bin/env bash
set -euo pipefail
MOTIOGEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MOTIOGEN_ROOT"
MOTIOGEN_BASELINE="${1:?baseline name required}"
MOTIOGEN_ENV="motiogen-${MOTIOGEN_BASELINE}"
MOTIOGEN_PYTHON=3.10
case "$MOTIOGEN_BASELINE" in
  motionlcm|omnicontrol|chois) ;;
  ardy|kimodo) MOTIOGEN_PYTHON=3.10 ;;
  *) exit 2 ;;
esac
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1 OMP_NUM_THREADS=1
# Never inherit desktop/session proxies for package downloads.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PIP_PROXY PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${MOTIOGEN_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT=30 PIP_RETRIES=2
if ! conda run -n "$MOTIOGEN_ENV" python -c 'import sys; print(sys.version)' >/dev/null 2>&1; then
  conda create -n "$MOTIOGEN_ENV" --override-channels -c https://repo.anaconda.com/pkgs/main "python=$MOTIOGEN_PYTHON" pip -y
fi
case "$MOTIOGEN_BASELINE" in
  motionlcm|omnicontrol|chois)
    MOTIOGEN_TORCH_WHEEL="$MOTIOGEN_ROOT/outputs/assets/wheels/torch-2.2.2+cu118-cp310-cp310-linux_x86_64.whl"
    if [[ -f "$MOTIOGEN_TORCH_WHEEL" ]]; then
      conda run --no-capture-output -n "$MOTIOGEN_ENV" python python/scripts/install_cached_torch.py "$MOTIOGEN_TORCH_WHEEL"
    else
      echo 'Missing cached CUDA torch wheel; register a verified mainland wheel source before continuing.' >&2
      exit 1
    fi
    MOTIOGEN_REQUIREMENTS=config/environments/legacy.txt
    if [[ "$MOTIOGEN_BASELINE" == motionlcm ]]; then
      MOTIOGEN_REQUIREMENTS=config/environments/motionlcm.txt
    fi
    conda run --no-capture-output -n "$MOTIOGEN_ENV" python -m pip install -r "$MOTIOGEN_REQUIREMENTS" --find-links outputs/assets/wheels
    ;;
  ardy|kimodo)
    MOTIOGEN_TORCH_WHEEL="$MOTIOGEN_ROOT/outputs/assets/wheels/torch-2.5.1+cu118-cp310-cp310-linux_x86_64.whl"
    conda run --no-capture-output -n "$MOTIOGEN_ENV" python python/scripts/install_cached_torch.py "$MOTIOGEN_TORCH_WHEEL"
    conda run --no-capture-output -n "$MOTIOGEN_ENV" python -m pip install numpy==1.26.4 scipy==1.11.2 psutil==6.1.1 matplotlib==3.7.5 --find-links outputs/assets/wheels
    conda run --no-capture-output -n motiogen-tools python python/scripts/stage_external_source.py "$MOTIOGEN_BASELINE"
    conda run --no-capture-output -n "$MOTIOGEN_ENV" python -m pip install -e "python/third_party/$MOTIOGEN_BASELINE" --find-links outputs/assets/wheels
    ;;
esac
if [[ "$MOTIOGEN_BASELINE" == omnicontrol || "$MOTIOGEN_BASELINE" == chois ]]; then
  mkdir -p python/third_party
  if [[ ! -d python/third_party/CLIP/.git ]]; then
    git clone https://github.com/openai/CLIP.git python/third_party/CLIP
  fi
  conda run --no-capture-output -n "$MOTIOGEN_ENV" python -m pip install -e python/third_party/CLIP --no-deps
fi
conda run --no-capture-output -n "$MOTIOGEN_ENV" python -c 'import fcntl,torch; h=open("develop_logs/test_runs/.gpu.lock","a"); fcntl.flock(h,fcntl.LOCK_EX); print(torch.__version__); print(torch.cuda.get_device_name(0)); x=torch.ones(32,device="cuda"); print((x@x).item())'
mkdir -p config/environments/locks
conda run --no-capture-output -n "$MOTIOGEN_ENV" python -m pip freeze > "config/environments/locks/$MOTIOGEN_BASELINE.txt"
