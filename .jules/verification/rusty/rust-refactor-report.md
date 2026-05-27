# Rusty Rust Refactor Report

## Repository Recon
Found multiple Python scripts inside `scripts/` doing various tasks:
- Hashing files (`verify-checksum-models.py`)
- Getting chat templates (`get_chat_template.py`)
- Generating markdown documentation (`create_ops_docs.py`)
- Parsing logs (`compare-logprobs.py`)
- Syncing vendor dependencies (`sync_vendor.py`)

A few scripts had already been ported to Rust inside the `rusty` workspace. `sync_vendor.py` was an excellent remaining candidate.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `scripts/sync_vendor.py` | Python 3 | Remove external python logic download, subprocess overhead, and simplify dependency | Low | Low | Selected |
| 2 | `scripts/compare-llama-bench.py` | Python 3 | High | High | High | Rejected |
| 3 | `scripts/server-bench.py` | Python 3 | High | High | High | Rejected |
| 4 | `scripts/tool_bench.py` | Python 3 | High | High | High | Rejected |
| 5 | `scripts/server-test-function-call.py` | Python 3 | Medium | High | High | Rejected |

## Selected Candidate

- Path: `scripts/sync_vendor.py`
- Current implementation: Python script using `urllib` and `subprocess` to call an external downloaded Python script `split.py`.
- Rust replacement: `rusty/src/bin/sync_vendor.rs` which fetches files via `reqwest` and implements the string splitting logic directly.
- Reason selected: Clean logic, removes the risk of arbitrary code execution from a dynamically downloaded python script, simplifies the deployment chain, and already fits the project's strategy of wrapping Python scripts around Rust implementations.

## Implementation Summary
Created `sync_vendor.rs` in `rusty/src/bin/` to download files via `reqwest::blocking::Client`. Extracted the header splitting logic from `split.py` and implemented it directly in Rust. Updated the `rusty/Cargo.toml` to include the `sync_vendor` binary. Replaced `scripts/sync_vendor.py` with a simple Python wrapper to compile and run the Rust binary.

## Before Benchmark
Run Time: 2961 ms

## After Benchmark
Run Time: 1787 ms

## Benchmark Delta
-39.6% speedup.

## Tests Run
`cargo check` passed.
Run of `sync_vendor.py` downloaded and successfully split `httplib.h`.

## Files Changed
- `scripts/sync_vendor.py`
- `rusty/Cargo.toml`
- `rusty/src/bin/sync_vendor.rs`

## Compatibility Notes
Works as a drop-in replacement. The Python wrapper mirrors the execution style of other Rust utilities in the codebase.

## Remaining Follow-Ups
None.
