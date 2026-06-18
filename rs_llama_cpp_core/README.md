# rs_llama_cpp_core

Safe RAII wrappers over `rs_llama_cpp_sys` for the ATOM-RS **Lane-A** engine — a pure-Rust shell driving the
in-tree llama.cpp 1-bit-turbo **HIP build** (gfx1030). No Python authored.

## API — embedding path (working)
- `LlamaBackend::init()` — process-wide one-time backend init (`AtomicBool` CAS singleton).
- `LlamaModel::load(&backend, path, n_gpu_layers)` — load a GGUF; `Drop` → `llama_model_free`.
- `model.embedding_context(n_ctx, Pooling::{Last,Mean,Cls,None,Unspecified})` — an embedding context that
  **borrows** the model (`LlamaContext<'a>` holds `&'a LlamaModel`), so it cannot outlive it and `Drop` order
  is reverse-of-init (context freed before model — required, since the context holds model pointers).
- `ctx.embed(text) -> Vec<f32>` — tokenize → `llama_decode` → pooled `llama_get_embeddings_seq` → L2-normalize.

Errors are a per-crate `thiserror` enum (`LlamaCppError`) with a `Result<T>` alias.

## Verified (2026-06-17, gfx1030 RX 6700 XT)
`cargo test -p rs_llama_cpp_core --features rocm` loads `Qwen3-Embedding-0.6B-Q8_0` on the GPU and produces a
**1024-dim, L2-normalized** embedding (`norm=1.00000`). This is the Lane-A serving keystone for the embedder
targets (lfm2-colbert / jina-code / qwen3-embedding).

## Runtime note
The shared libs (`libllama.so`, `libggml-*.so`) live in `build-hip/bin`. `rs_llama_cpp_sys/build.rs` emits an
rpath for its **own** artifacts, but cargo does not propagate it to downstream test/bin binaries — run with
`LD_LIBRARY_PATH=<repo>/build-hip/bin` (the eventual server binary sets this, or add a workspace
`.cargo/config.toml` rpath). Also export `HSA_OVERRIDE_GFX_VERSION=10.3.0` for gfx1030.

## Next
- ColBERT multi-vector path (`Pooling::None` + per-token `llama_get_embeddings_ith`).
- Generative/completion path (sampler + token loop, KV cache).
- Wrap in the axum OpenAI server (`/v1/embeddings` + `/v1/completions`); repoint embed-pool.
