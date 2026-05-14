# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `scripts/verify-checksum-models.py` | Python | Faster execution through parallel hashing, less memory usage overhead, compiled binary | Low | Low | Selected |
| 2 | `scripts/compare-llama-bench.py` | Python | Faster log parsing/summarizing | Medium | Low | Not Selected |
| 3 | `scripts/compare-logprobs.py` | Python | Faster request processing/JSON parsing | Medium | Low | Not Selected |
| 4 | `scripts/create_ops_docs.py` | Python | Better file traversing and markdown generation | Low | Low | Not Selected |
| 5 | `scripts/tool_bench.py` | Python | More robust concurrent execution, memory safety | Medium | Medium | Not Selected |

## Selected Candidate

- Path: `scripts/verify-checksum-models.py` -> `rusty/src/main.rs`
- Current implementation: Single-threaded Python script that iterates over files listed in `SHA256SUMS` and calculates the SHA256 checksum sequentially, loading 16MB blocks at a time.
- Rust replacement: Pure Rust CLI (`rusty` crate) using `clap` for arg parsing, `rayon` for concurrent parallel hashing, and `sha2` for fast Rust-native SHA256 calculation.
- Reason selected: File hashing is an extremely CPU-bound and I/O parallelizable task. The Python implementation is completely sequential. Replacing this with a simple parallel Rust pipeline offers a massive performance improvement for verifying numerous multi-gigabyte models without introducing broad architectural risk or new dependencies to the core C++ codebase. It serves as a drop-in replacement that takes advantage of all available CPU cores.

## Implementation Summary

I created a new pure Rust crate inside a `rusty` workspace. It parses `SHA256SUMS` (the same expected format), checks existence, and distributes the SHA256 checksum calculations across a Rayon parallel thread pool. Finally, it outputs the identical report table as the Python script.

## Before Benchmark
`python3 scripts/verify-checksum-models.py`
Time (mean ± σ):      7.477 s ±  0.182 s

## After Benchmark
`./rusty/target/release/rusty`
Time (mean ± σ):      3.609 s ±  0.045 s

## Benchmark Delta
- Before timing: 7.477 s
- After timing: 3.609 s
- Percent change: -51.7% (2.07x faster)
- Notes: The benchmark hashed 20 files of 100MB each (2GB total). The speedup would be even greater on a machine with more CPU cores or larger files (models are typically multiple GBs).

## Tests Run
- Compiled Rust code: `cargo build --release` (Passed)
- Checked output parity: `diff <(python3 ... ) <(./rusty...)` (Passed - output identical except the Rust script interleaves stderr progress faster)
- Hyperfine benchmark suite: Checked both Python and Rust on identical synthetic payloads.

## Files Changed
- Created `rusty/Cargo.toml`
- Created `rusty/src/main.rs`

## Compatibility Notes
The output table headers and rows are character-for-character identical to the Python output script.

## Remaining Follow-Ups
- Could wire up the Makefile or CMake to build the `rusty` tool by default or as an optional utility.
- The Python script is left as a fallback; could be entirely removed once CI explicitly validates the Rust script in a matrix.
