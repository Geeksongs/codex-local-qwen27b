#!/bin/bash
# H200 adaptation of the original 24GB-card llama-server.sh.
# See HANDOVER-H200.md for the measurements behind every flag choice below.
set -euo pipefail

REQUIRED_CTX=262144
LUCEBOX_DIR="${LUCEBOX_DIR:-/workspace/python_song/lucebox/server}"
MODELS_DIR="${MODELS_DIR:-/workspace/python_song/models}"
LOG_DIR="${LOG_DIR:-/workspace/python_song/logs}"
mkdir -p "${LOG_DIR}"

cd "${LUCEBOX_DIR}"

./build/dflash_server "${MODELS_DIR}/Qwen3.8-27B-Uncensored-Q8_0.gguf" \
  --draft "${MODELS_DIR}/draft/qwen38-dflash2-q8_0.gguf" \
  --target-device cuda:0 --draft-device cuda:0 \
  --draft-block-size 16 \
  --max-ctx "${REQUIRED_CTX}" \
  --kvflash auto \
  --prefill-drafter "${MODELS_DIR}/drafter/Qwen3-0.6B-Q8_0.gguf" \
  --ddtree --ddtree-budget 24 \
  --chunk 1024 \
  --cache-type-k f16 --cache-type-v f16 \
  --prefix-cache-slots 32 \
  --prefill-cache-slots 16 \
  --host 127.0.0.1 --port 18081 \
  >> "${LOG_DIR}/dflash_server.log" 2>&1 &
SERVER_PID=$!

trap 'kill -9 "${SERVER_PID}" 2>/dev/null' EXIT TERM INT

for i in $(seq 1 60); do
  ACTUAL_CTX=$(curl -s --max-time 2 "http://127.0.0.1:18081/v1/models" 2>/dev/null \
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
