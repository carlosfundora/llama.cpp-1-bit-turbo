# Benchmark Summary

- Before Command: `python scripts/get_chat_template.py microsoft/Phi-3.5-mini-instruct`
- After Command: `rusty/target/release/get_chat_template microsoft/Phi-3.5-mini-instruct`
- Before Duration: 705 ms
- After Duration: 224 ms
- Percent Change: -68.2% (approx 3x faster)

Notes: The Rust version is significantly faster as it avoids python interpreter initialization, avoids loading `requests` / `huggingface_hub` and `urllib3` modules, and leverages compiled execution for JSON parsing and regex.
