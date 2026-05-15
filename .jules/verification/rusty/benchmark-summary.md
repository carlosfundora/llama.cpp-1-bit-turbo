# Benchmark Summary

* **Before command**: `python3 scripts/compare-logprobs.py compare dummy1.log dummy2.log output.md`
* **After command**: `./rusty/target/release/compare_logprobs compare dummy1.log dummy2.log output.md`
* **Before timing**: ~1418.88 ms
* **After timing**: ~319.16 ms
* **Percent change**: -77.5% time reduction (approx 4.4x faster)
* **Notes**: Both benchmarks parsed the same 50,000 line JSON log datasets. The Python version was fully functional and single-threaded. The Rust refactor achieves significant speedups primarily through faster JSON parsing (`serde_json`) and static typing layout optimizations.
