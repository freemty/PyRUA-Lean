#!/usr/bin/env bash
# usage: launch.sh <libero|robotwin> <cuda-devices> <out> [run_sweep.py args...]
# Runs run_sweep.py on this node with the fs-mbz environment; LIBERO renders with Mesa through EGL (many
# renderers per card, and no NVIDIA-EGL next to SAPIEN).
set -euo pipefail
KIND=$1 DEVICES=$2 OUT=$3; shift 3
source /mnt/weka/shrd/hao/yuanbo/b2/pyrua-lean/env.sh
export OPENAI_API_KEY=none PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/tmp/pyrua-$KIND-$USER; mkdir -p "$TMPDIR"
if [ "$KIND" = libero ]; then
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
  export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2
fi
cd /mnt/weka/home/hao.zhang/yuanbo/pyrua-lean
exec python repro/run_sweep.py --cells "${CELLS:-repro/sweep-130.json}" --out "$OUT" --base-url http://127.0.0.1:30851/v1 \
  --only "$KIND" --cuda-devices "$DEVICES" "$@"
