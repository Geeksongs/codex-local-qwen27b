#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

REQUIRED_CTX=262144
cd /workspace/lucebox/server

./build/dflash_server /workspace/models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf \
  --draft /workspace/models/qwen38-dflash2-q8_0.gguf \
  --target-device cuda:0 --draft-device cuda:0 \
  --draft-block-size 8 \
  --max-ctx "${REQUIRED_CTX}" \
  --think-max-tokens 14336 \
  --kvflash auto \
  --prefill-drafter /workspace/models/drafter/Qwen3-0.6B-Q8_0.gguf \
  --ddtree --ddtree-budget 22 \
  --cache-type-k f16 --cache-type-v f16 \
  --prefix-cache-slots 32 \
  --prefill-cache-slots 16 \
  --host 127.0.0.1 --port 18081 &
SERVER_PID=$!

# Bash does not forward signals to a backgrounded child by default, so a
# supervisor stop/restart (SIGTERM to this script) would otherwise leave
# dflash_server running as an orphan holding VRAM -- causing the *next*
# start attempt to OOM against a "ghost" instance. Make sure it dies with us.
trap 'kill -9 "${SERVER_PID}" 2>/dev/null' EXIT TERM INT

# Hard guard: this deployment must always serve the model's full 262144-token
# context. Refuse to keep running at anything else instead of silently
# degrading (the engine itself has no silent-downgrade path for --max-ctx --
# a real mismatch is always a hard crash -- this also catches a hung/partial
# load or an accidental future edit to the flag above).
for i in $(seq 1 60); do
  ACTUAL_CTX=$(curl -s --max-time 2 "http://127.0.0.1:18081/v1/models" 2>/dev/null \
    | grep -o '"context_length":[0-9]*' | head -1 | grep -o '[0-9]*$')
  if [ -n "${ACTUAL_CTX}" ]; then
    if [ "${ACTUAL_CTX}" != "${REQUIRED_CTX}" ]; then
      echo "FATAL: server reports context_length=${ACTUAL_CTX}, required ${REQUIRED_CTX}. Refusing to run at a reduced context." >&2
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
