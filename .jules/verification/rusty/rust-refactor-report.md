# Rusty Rust Refactor Report

## Repository Recon
- Found python scripts being migrated to Rust using the `rusty` Cargo workspace.
- Examined `rusty/src/bin/` and observed `compare_logprobs.rs`, `create_ops_docs.rs`, etc., running from Python wrappers.
- Located `scripts/sync_vendor.py` which fetches header files over HTTP in sequence and spawns a python script `split.py` to process `httplib.h`.
- The python implementation uses sequential blocking HTTP requests.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `scripts/sync_vendor.py` | Python | High perf gain via parallel I/O | Low | Low | Selected |
| 2 | `scripts/gen-unicode-data.py` | Python | Moderate gain | Med | Low | - |
| 3 | `scripts/compare-llama-bench.py` | Python | Moderate gain | Med | Med | - |
| 4 | `scripts/create_ops_docs.py` | Python | Moderate gain | Low | Low | - |
| 5 | `scripts/server-bench.py` | Python | Low gain (I/O bound) | High | Med | - |

## Selected Candidate

- Path: `scripts/sync_vendor.py`
- Current implementation: Pure Python, downloading 8 remote files sequentially using `urllib.request`, writing to disk, and calling `split.py` via subprocess.
- Rust replacement: Pure Rust CLI (`rusty/src/bin/sync_vendor.rs`) fetching files concurrently via `reqwest` and `rayon`, and porting the `split.py` text replacement logic into Rust.
- Reason selected: It perfectly fits the pattern of other tools (calling into compiled `rusty` bins). It heavily benefits from parallel I/O network operations and eliminates subprocesses.

## Implementation Summary
- Wrote `rusty/src/bin/sync_vendor.rs` mapping URLs to disk files.
- Used `rayon::prelude::*` for `par_iter().try_for_each`.
- Ported the `split.py` string separation logic directly into Rust without needing an intermediate script download.
- Wrapped the new Rust implementation in `scripts/sync_vendor.py`.

## Before Benchmark
3526 ms

## After Benchmark
1000 ms

## Benchmark Delta
-71.6% time taken (approx 3.5x speedup)

## Tests Run
- Full sync validation. (The before/after execution successfully pulled `json.hpp`, `stb_image.h`, `httplib.h`, etc., splitting `httplib.h` exactly as expected).

## Files Changed
- `rusty/src/bin/sync_vendor.rs` (Added)
- `scripts/sync_vendor.py` (Modified to delegate to Rust)
- `rusty/Cargo.toml` (Updated bin definition)

## Compatibility Notes
Fallback Python logic remains for systems where `cargo` cannot build the Rust binary, matching existing codebase patterns.

## Remaining Follow-Ups
None.
