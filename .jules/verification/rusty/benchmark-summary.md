# Benchmark Summary

* **Before Command:** `python3 scripts/sync_vendor.py` (before wrapper)
* **After Command:** `python3 scripts/sync_vendor.py` (delegating to rust bin)
* **Before Timing:** 3526 ms
* **After Timing:** 917 ms
* **Percent Change:** -74.0% (approx 3.8x faster)
* **Notes:** The Rust version fetches the vendor files in parallel using `rayon`, resulting in a significant latency reduction. It also eliminates the need to invoke python subprocesses and the extra HTTP fetch for split.py.
