#!/bin/bash
set -e

export MODEL_PATH="/home/local/ai/models/registry/Unsloth/LFM2-2.6B-Exp-GGUF/Q8/LFM2-2.6B-Exp-Q8_0.gguf"
export HOST="0.0.0.0"
export PORT="${PORT:-9960}"
export THREADS="${THREADS:-12}"

# CPU-only inference (llama.cpp 1-bit turbo on CPU is respectable ~500 t/s)
exec /home/local/ai/engines/llama.cpp-1-bit-turbo/build/bin/llama-server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS" \
  --verbose
