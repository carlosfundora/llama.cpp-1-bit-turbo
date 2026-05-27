# Benchmark Summary

- Before Command: `python3 scripts/sync_vendor.py` (Old implementation using `urllib` and executing an external Python script `split.py`)
- After Command: `python3 scripts/sync_vendor.py` (New implementation passing to compiled Rust binary `sync_vendor` using `reqwest` and native header splitting)

- Before Timing: 2961 ms
- After Timing: 1787 ms

- Percent Change: -39.6% (Faster)

- Notes: The Rust implementation removes the overhead of downloading and calling an external Python script (`split.py`) as a subprocess, resulting in a cleaner and faster execution while maintaining the same behavior.
