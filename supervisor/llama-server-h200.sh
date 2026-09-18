#!/bin/bash
# H200 adaptation of the original 24GB-card llama-server.sh.
# See HANDOVER-H200.md for the full measurement log behind every choice here,
# including why this diverges substantially from the 262144/KVFlash config
# this repo originally shipped: KVFlash's bounded pool was found to silently
# corrupt long-context retrieval on this hardware (0/5 needle recall past its
# 16384-token pool in testing), so this deployment runs a full resident KV
# cache instead, extended past the model's native 262144 tokens via YaRN.
set -euo pipefail

REQUIRED_CTX=1572864   # 262144 * 6 (--yarn-factor 6.0) -- see ceiling map below
LUCEBOX_DIR="${LUCEBOX_DIR:-/workspace/python_song/lucebox/server}"
MODELS_DIR="${MODELS_DIR:-/workspace/python_song/models}"
LOG_DIR="${LOG_DIR:-/workspace/python_song/logs}"
mkdir -p "${LOG_DIR}"

cd "${LUCEBOX_DIR}"

# GPU3: empirically the least-loaded of the 4 shared H200s during tuning.
# Re-check `nvidia-smi` before relying on this — other tenants' usage on this
# box moves by tens of GB over time. VRAM ceiling map (this model, q8_0 KV,
# no KVFlash, measured against a realistic ~25K-token prompt, not just a
# toy short one):
#   1.0M (factor=4) -> ~39 GB free after a real request
#   1.5M (factor=6) -> ~17 GB free after a real request  <- deployed here
#   1.75M (factor=7) -> ~8.6 GB free -- works but thin margin, not chosen
#   2.0M (factor=8) -> OOM on the per-request rollback-cache allocation
./build/dflash_server "${MODELS_DIR}/Qwen3.8-27B-Uncensored-Q8_0.gguf" \
  --draft "${MODELS_DIR}/draft/qwen38-dflash2-q8_0.gguf" \
  --target-device cuda:3 --draft-device cuda:3 \
  --draft-block-size 16 \
  --max-ctx "${REQUIRED_CTX}" \
  --yarn-factor 6.0 --yarn-orig-ctx 262144 \
  --ddtree --ddtree-budget 24 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 127.0.0.1 --port 18097 \
  >> "${LOG_DIR}/dflash_server.log" 2>&1 &
SERVER_PID=$!

trap 'kill -9 "${SERVER_PID}" 2>/dev/null' EXIT TERM INT

for i in $(seq 1 90); do
  ACTUAL_CTX=$(curl -s --max-time 2 "http://127.0.0.1:18097/v1/models" 2>/dev/null \
    | grep -o '"context_length":[0-9]*' | head -1 | grep -o '[0-9]*$')
  if [ -n "${ACTUAL_CTX}" ]; then
    if [ "${ACTUAL_CTX}" != "${REQUIRED_CTX}" ]; then
      echo "FATAL: server reports context_length=${ACTUAL_CTX}, required ${REQUIRED_CTX}." >&2
      kill -9 "${SERVER_PID}" 2>/dev/null
      exit 1
    fi
    echo "context_length verified: ${ACTUAL_CTX}"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FATAL: server process exited before reporting context_length" >&2
    exit 1
  fi
  sleep 2
done

wait "${SERVER_PID}"
