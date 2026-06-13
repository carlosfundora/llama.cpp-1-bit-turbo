#!/bin/bash
echo 'Benchmarking Python get_chat_template.py'

start=$(date +%s%3N)
python scripts/get_chat_template.py microsoft/Phi-3.5-mini-instruct > /dev/null
end=$(date +%s%3N)
dur=$((end-start))

cat << JSON > .jules/verification/rusty/before-benchmark.json
{
  "candidate": "scripts/get_chat_template.py",
  "implementation": "before",
  "command": "python scripts/get_chat_template.py microsoft/Phi-3.5-mini-instruct",
  "timestamp": "$(date -Iseconds)",
  "iterations": 1,
  "input_description": "microsoft/Phi-3.5-mini-instruct template",
  "duration_ms": $dur
}
JSON
