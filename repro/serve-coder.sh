#!/usr/bin/env bash
# Serve PyRUA-Lean Coder v0.1-4B on one card with vLLM (the model card uses SGLang; same parsers, 65,536 context).
# usage: serve-coder.sh <card> [port]
set -euo pipefail
CARD=$1 PORT=${2:-30851}
MODEL=/mnt/weka/home/hao.zhang/yuanbo/models/pyrua-lean/coder-4b
V=/mnt/weka/shrd/hao/yuanbo/b2/envs/vllm-0.32-venv
m=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$m" -lt 2048 ] || { echo "refusing: card $CARD has ${m} MiB in use"; exit 1; }
export CUDA_VISIBLE_DEVICES=$CARD PATH=$V/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export HF_HUB_OFFLINE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 TMPDIR=/tmp/pyrua-coder-$USER; mkdir -p $TMPDIR
echo "== $(hostname) $(date -u +%FT%TZ) coder-4b card $CARD port $PORT"
exec "$V/bin/vllm" serve "$MODEL" --served-model-name DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --max-model-len 65536 \
  --gpu-memory-utilization ${UTIL:-0.80} --enable-prefix-caching --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image": 64, "video": 0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
