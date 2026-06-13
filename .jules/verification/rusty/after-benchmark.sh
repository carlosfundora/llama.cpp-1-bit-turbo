#!/bin/bash
echo 'Benchmarking Rust get_chat_template'

start=$(date +%s%3N)
rusty/target/release/get_chat_template microsoft/Phi-3.5-mini-instruct > /dev/null
end=$(date +%s%3N)
dur=$((end-start))

cat << JSON > .jules/verification/rusty/after-benchmark.json
{
  "candidate": "scripts/get_chat_template.py",
  "implementation": "after",
  "command": "rusty/target/release/get_chat_template microsoft/Phi-3.5-mini-instruct",
  "timestamp": "$(date -Iseconds)",
  "iterations": 1,
  "input_description": "microsoft/Phi-3.5-mini-instruct template",
  "duration_ms": $dur
}
JSON
