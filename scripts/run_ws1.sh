#!/usr/bin/env bash
# Train + evaluate on a CUDA workstation (headless, EGL rendering).
#
# Usage (from anywhere):
#   scripts/run_ws1.sh smoke [TASK ...]   # ~minutes per task; pipeline check only
#   scripts/run_ws1.sh full  [TASK ...]   # real budgets from humanoid_mimic_rl/configs.py
# TASK defaults to: walk getup backflip mimic supermimic
# ("supermimic" = the mimic task trained with train_supermimic.py instead of PPO)
#
# Environment variables:
#   VENV   python venv to activate          (default: $HOME/.venvs/mjrl)
#   IMPL   MJX implementation, jax | warp   (default: jax)
#   LOGDIR run output directory             (default: <repo>/runs)
#   EXTRA  extra args for train.py / train_supermimic.py
#          (e.g. "--num-timesteps 50000000" or, for supermimic, "--iterations 8000")
#
# Every task's training and evaluation output is printed line by line and
# tee'd to $LOGDIR/<profile>_<task>_<timestamp>.log. The script stops at the
# first failing command.
set -euo pipefail

PROFILE="${1:?usage: $0 smoke|full [TASK ...]}"
shift
case "$PROFILE" in smoke|full) ;; *) echo "profile must be smoke or full" >&2; exit 2 ;; esac
TASKS=("$@")
[[ ${#TASKS[@]} -eq 0 ]] && TASKS=(walk getup backflip mimic supermimic)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/.venvs/mjrl}"
IMPL="${IMPL:-jax}"
LOGDIR="${LOGDIR:-$REPO/runs}"
EXTRA="${EXTRA:-}"
mkdir -p "$LOGDIR"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO"

echo "== $(date -Is) repo=$REPO profile=$PROFILE impl=$IMPL tasks=${TASKS[*]}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python - <<'EOF'
import importlib.metadata as md, jax
for p in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "playground", "brax", "warp-lang", "mediapy"):
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: NOT INSTALLED")
print("jax devices:", jax.devices())
EOF

if python -c "import pytest" 2>/dev/null; then
  python -m pytest -q tests
else
  echo "WARNING: pytest not installed in $VENV; skipping unit tests (pip install pytest)"
fi

REF="$REPO/data/generated/t1_backflip_keyframed.npz"
for task in "${TASKS[@]}"; do
  if [[ ( "$task" == mimic || "$task" == supermimic ) && ! -f "$REF" ]]; then
    echo "== generating reference clip $REF"
    python scripts/make_reference.py --out "$REF"
    python eval_render.py --reference-only "$REF" --render-every 2
  fi
done

for task in "${TASKS[@]}"; do
  stamp="$(date +%Y%m%d-%H%M%S)"
  log="$LOGDIR/${PROFILE}_${task}_${stamp}.log"
  echo "== $(date -Is) train $task ($PROFILE) -> $log"
  if [[ "$task" == supermimic ]]; then
    # shellcheck disable=SC2086
    python train_supermimic.py --profile "$PROFILE" --impl "$IMPL" --logdir "$LOGDIR" $EXTRA 2>&1 | tee "$log"
  else
    # shellcheck disable=SC2086
    python train.py --task "$task" --profile "$PROFILE" --impl "$IMPL" --logdir "$LOGDIR" $EXTRA 2>&1 | tee "$log"
  fi
  run_dir="$(grep -m1 '^run dir: ' "$log" | sed 's/^run dir: //')"
  if [[ "$PROFILE" == smoke ]]; then n_ep=16; else n_ep=128; fi
  speed=1.0
  [[ "$task" == backflip || "$task" == mimic || "$task" == supermimic ]] && speed=0.5
  echo "== $(date -Is) eval $task -> $run_dir/eval"
  python eval_render.py --run-dir "$run_dir" --num-episodes "$n_ep" --video-episodes 2 \
    --playback-speed "$speed" 2>&1 | tee -a "$log"
done
echo "== $(date -Is) all done"
