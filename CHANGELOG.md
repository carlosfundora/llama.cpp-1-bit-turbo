## [2026-05-13]
### Maintenance
- Renamed `common_init_sampler_from_model` to `common_params_sampling_init` and moved it to `common/sampling.cpp`.

## Documentation
- None required

## [2026-05-11]
### Maintenance
- Resolved merge conflicts for PR #33 (`fix/lfm2-colbert-embeddings`).
- Successfully merged `master` into `pr-33` using `--allow-unrelated-histories` and a `theirs` resolution strategy.
- Verified fix presence in `src/models/lfm2.cpp` and confirmed ROCm build stability (gfx1030).
