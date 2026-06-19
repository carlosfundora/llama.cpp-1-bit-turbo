//! Safe RAII wrappers over [`rs_llama_cpp_sys`] for the ATOM-RS Lane-A engine.
//!
//! - [`LlamaBackend`] — process-wide one-time backend init (CAS singleton).
//! - [`LlamaModel`] — a loaded GGUF model; `Drop` calls `llama_model_free`.
//! - [`LlamaContext`] — an inference context that **borrows** its model (`&'a LlamaModel`), so the
//!   borrow checker forbids it outliving the model and `Drop` order is reverse-of-init
//!   (context freed before model — required, since the context holds model pointers).
//!
//! This first cut covers the **embedding path** (pooled sequence embedding) used to serve
//! lfm2-colbert / jina-code / qwen3-embedding on the gfx1030 HIP base.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use rs_llama_cpp_sys as sys;

pub type Result<T> = std::result::Result<T, LlamaCppError>;

#[derive(Debug, thiserror::Error)]
pub enum LlamaCppError {
    #[error("model path contains an interior NUL byte: {0:?}")]
    BadPath(String),
    #[error("failed to load model from {0}")]
    ModelLoad(String),
    #[error("failed to create llama context")]
    ContextInit,
    #[error("tokenization failed (llama_tokenize returned {0})")]
    Tokenize(i32),
    #[error("empty input produced no tokens")]
    EmptyInput,
    #[error("llama_decode failed (code {0})")]
    Decode(i32),
    #[error("no sequence embeddings returned (pooling disabled or seq mismatch)")]
    NoEmbeddings,
}

/// Pooling strategy for a sequence embedding. `Unspecified` keeps the model/GGUF default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
}

impl Pooling {
    /// `None` => leave the context default (Unspecified); `Some(raw)` => set it explicitly.
    fn raw_opt(self) -> Option<sys::llama_pooling_type> {
        match self {
            Pooling::Unspecified => None,
            Pooling::None => Some(sys::LLAMA_POOLING_TYPE_NONE),
            Pooling::Mean => Some(sys::LLAMA_POOLING_TYPE_MEAN),
            Pooling::Cls => Some(sys::LLAMA_POOLING_TYPE_CLS),
            Pooling::Last => Some(sys::LLAMA_POOLING_TYPE_LAST),
        }
    }
}

/// Process-wide guard: `llama_backend_init` runs exactly once.
static BACKEND_INIT: AtomicBool = AtomicBool::new(false);

/// Proof token that the llama.cpp backend has been initialized. Cheap to clone-by-re-init.
#[derive(Debug)]
pub struct LlamaBackend {
    _private: (),
}

impl LlamaBackend {
    /// Initialize the global llama.cpp backend (idempotent across the process).
    pub fn init() -> Self {
        if BACKEND_INIT
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // SAFETY: guarded by the CAS above so this runs exactly once; `llama_backend_init`
            // is the documented global init and takes no arguments.
            unsafe { sys::llama_backend_init() };
        }
        Self { _private: () }
    }
}
// NOTE: deliberately no `llama_backend_free` on drop — the backend is process-global infra and
// freeing it while any model/context may still exist is unsound.

/// A loaded GGUF model.
pub struct LlamaModel {
    ptr: NonNull<sys::llama_model>,
}

impl LlamaModel {
    /// Load a GGUF model, offloading `n_gpu_layers` to the GPU (use a large value, e.g. 999, for full GPU).
    pub fn load(_backend: &LlamaBackend, path: &str, n_gpu_layers: i32) -> Result<Self> {
        let cpath = CString::new(path).map_err(|_| LlamaCppError::BadPath(path.to_string()))?;
        // SAFETY: default params are a plain POD struct; we only set n_gpu_layers.
        let mut mparams = unsafe { sys::llama_model_default_params() };
        mparams.n_gpu_layers = n_gpu_layers;
        // SAFETY: cpath is a valid NUL-terminated C string for the duration of the call.
        let raw = unsafe { sys::llama_model_load_from_file(cpath.as_ptr(), mparams) };
        let ptr = NonNull::new(raw).ok_or_else(|| LlamaCppError::ModelLoad(path.to_string()))?;
        Ok(Self { ptr })
    }

    /// Embedding dimensionality (`n_embd`) of the model.
    pub fn n_embd(&self) -> i32 {
        // SAFETY: self.ptr is a live model for the lifetime of `self`.
        unsafe { sys::llama_model_n_embd(self.ptr.as_ptr()) }
    }

    /// Output embedding dimensionality after any in-graph dense head (`n_embd_out`). Equals
    /// [`n_embd`] for plain embedders; for the LFM2-ColBERT GGUF it is the projected ColBERT dim
    /// (128 vs the 1024 backbone) — the per-token vectors from [`LlamaContext::embed_tokens`] have
    /// this width.
    pub fn n_embd_out(&self) -> i32 {
        // SAFETY: live model.
        unsafe { sys::llama_model_n_embd_out(self.ptr.as_ptr()) }
    }

    fn vocab(&self) -> *const sys::llama_vocab {
        // SAFETY: live model.
        unsafe { sys::llama_model_get_vocab(self.ptr.as_ptr()) }
    }

    /// Create an embedding context that borrows this model (so it cannot outlive it).
    pub fn embedding_context(&self, n_ctx: u32, pooling: Pooling) -> Result<LlamaContext<'_>> {
        // SAFETY: default params are POD; we set n_ctx/embeddings/pooling only.
        let mut cparams = unsafe { sys::llama_context_default_params() };
        cparams.n_ctx = n_ctx;
        cparams.embeddings = true;
        if let Some(pt) = pooling.raw_opt() {
            cparams.pooling_type = pt;
        }
        // SAFETY: self.ptr is a live model; init_from_model returns null on failure (checked).
        let raw = unsafe { sys::llama_init_from_model(self.ptr.as_ptr(), cparams) };
        let ptr = NonNull::new(raw).ok_or(LlamaCppError::ContextInit)?;
        Ok(LlamaContext { ptr, model: self })
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        // SAFETY: ptr was obtained from llama_model_load_from_file and is freed exactly once here.
        unsafe { sys::llama_model_free(self.ptr.as_ptr()) };
    }
}

/// An inference context bound to (and outlived by) its [`LlamaModel`].
pub struct LlamaContext<'a> {
    ptr: NonNull<sys::llama_context>,
    model: &'a LlamaModel,
}

impl LlamaContext<'_> {
    /// Tokenize + decode + extract the pooled sequence embedding as **RAW fp32** (NO normalization,
    /// full `n_embd`).
    ///
    /// This is the Matryoshka-safe primitive: a caller that wants a reduced dimensionality MUST
    /// truncate this raw vector to the target dim and only THEN L2-normalize (see [`embed_dim`]).
    /// Normalizing the full vector first and slicing afterwards yields a non-unit, mis-scaled
    /// vector — the ATOM reference (`model_runner.py:1817-1819`) slices first, normalizes second.
    pub fn embed_raw(&mut self, text: &str) -> Result<Vec<f32>> {
        // Each embedding is an independent forward pass. Clear any KV/memory left from a previous
        // embed on this (persistent, server-reused) context so sequences don't bleed together via
        // accumulated positions/attention. A fresh context is a no-op clear.
        // SAFETY: ctx is live; llama_get_memory + llama_memory_clear are the documented reset path.
        unsafe {
            let mem = sys::llama_get_memory(self.ptr.as_ptr());
            sys::llama_memory_clear(mem, true);
        }
        let mut tokens = self.tokenize(text, true)?;
        if tokens.is_empty() {
            return Err(LlamaCppError::EmptyInput);
        }
        // SAFETY: batch_get_one borrows the token slice for the duration of the decode call below;
        // `tokens` outlives that call.
        let batch = unsafe { sys::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        // SAFETY: ctx is live; batch references valid memory that lives across this call.
        let rc = unsafe { sys::llama_decode(self.ptr.as_ptr(), batch) };
        if rc < 0 {
            return Err(LlamaCppError::Decode(rc));
        }
        let n_embd = self.model.n_embd() as usize;
        // SAFETY: ctx is live; seq 0 is the single decoded sequence. Returns null if no pooled embd.
        let eptr = unsafe { sys::llama_get_embeddings_seq(self.ptr.as_ptr(), 0) };
        if eptr.is_null() {
            return Err(LlamaCppError::NoEmbeddings);
        }
        // SAFETY: the pooled embedding for seq 0 is `n_embd` contiguous f32 owned by the ctx.
        Ok(unsafe { std::slice::from_raw_parts(eptr, n_embd) }.to_vec())
    }

    /// Tokenize + decode + extract the pooled sequence embedding, L2-normalized (full `n_embd`).
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed_raw(text)?;
        l2_normalize(&mut v);
        Ok(v)
    }

    /// Matryoshka embedding: pool (fp32) → **truncate to `dim` (prefix slice)** → **L2-normalize the
    /// slice**, in that exact order. `dim == None` returns the full vector L2-normalized (identical to
    /// [`embed`]). Matches the ATOM reference order byte-for-byte (`model_runner.py:1817-1819`:
    /// `pooled = pooled[:dimensions]; F.normalize(pooled, p=2)`).
    pub fn embed_dim(&mut self, text: &str, dim: Option<usize>) -> Result<Vec<f32>> {
        Ok(matryoshka_truncate_normalize(self.embed_raw(text)?, dim))
    }

    /// Number of tokens `text` produces with the same `add_special=true` the embedding path uses.
    /// Used to populate the OpenAI `usage.prompt_tokens` faithfully (ATOM sets
    /// `prompt_tokens == total_tokens == sum of per-input prompt tokens`).
    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self.tokenize(text, true)?.len())
    }

    /// **Per-token (unpooled) embeddings** for ColBERT-style multi-vector retrieval. Returns one
    /// row of width [`LlamaModel::n_embd_out`] per token (the in-graph dense head is applied under
    /// `pooling = none`). The vectors are **raw** (NOT normalized) — the ColBERT path L2-normalizes
    /// each token row itself before MaxSim.
    ///
    /// REQUIRES the context to have been created with [`Pooling::None`]; otherwise the engine pools
    /// and `llama_get_embeddings_ith` is not the per-token output. Returns the rows in token order.
    pub fn embed_tokens(&mut self, text: &str) -> Result<Vec<Vec<f32>>> {
        // Independent forward pass — clear any prior KV/memory on this reused context.
        // SAFETY: ctx is live; documented reset path.
        unsafe {
            let mem = sys::llama_get_memory(self.ptr.as_ptr());
            sys::llama_memory_clear(mem, true);
        }
        let tokens = self.tokenize(text, true)?;
        if tokens.is_empty() {
            return Err(LlamaCppError::EmptyInput);
        }
        let n = tokens.len();
        // logits[i] = 1 for EVERY token so each position's embedding is produced under pooling=none.
        let mut batch = LlamaBatch::with_capacity(n, 1);
        for (i, &tok) in tokens.iter().enumerate() {
            batch.add(tok, i as sys::llama_pos, 0, true);
        }
        // SAFETY: ctx is live; batch arrays live until `batch` drops, after this call returns.
        let rc = unsafe { sys::llama_decode(self.ptr.as_ptr(), batch.raw) };
        if rc < 0 {
            return Err(LlamaCppError::Decode(rc));
        }
        let n_out = self.model.n_embd_out() as usize;
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: ctx is live; under pooling=none, ith returns the i-th flagged token's row of
            // `n_out` contiguous f32 owned by the ctx. Null => not produced (treated as error).
            let eptr = unsafe { sys::llama_get_embeddings_ith(self.ptr.as_ptr(), i as i32) };
            if eptr.is_null() {
                return Err(LlamaCppError::NoEmbeddings);
            }
            rows.push(unsafe { std::slice::from_raw_parts(eptr, n_out) }.to_vec());
        }
        Ok(rows)
    }

    /// **ColBERT token preparation** for MaxSim reranking. Tokenizes WITH special tokens (so content
    /// tokens attend to BOS/EOS in the forward pass, like the ATOM reference), truncates to
    /// `max_tokens`, runs the per-token forward (pooling=none), then for SCORING **drops control /
    /// special tokens** (`llama_vocab_is_control`, == ATOM's `~special_tokens_mask`) and
    /// **L2-normalizes each kept row** (so a later dot product is cosine). Returns the kept rows
    /// `[n_kept, n_embd_out]` in token order. REQUIRES a [`Pooling::None`] context.
    ///
    /// Caller supplies the role prefix (`"[Q] "` / `"[D] "`) in `text`. ATOM masks ONLY special
    /// tokens — it does NOT apply PyLate skiplist/query-expansion, and neither does this.
    pub fn colbert_tokens(&mut self, text: &str, max_tokens: Option<usize>) -> Result<Vec<Vec<f32>>> {
        // SAFETY: ctx is live; documented KV reset for an independent forward.
        unsafe {
            let mem = sys::llama_get_memory(self.ptr.as_ptr());
            sys::llama_memory_clear(mem, true);
        }
        let mut tokens = self.tokenize(text, true)?;
        if let Some(m) = max_tokens {
            tokens.truncate(m);
        }
        if tokens.is_empty() {
            return Err(LlamaCppError::EmptyInput);
        }
        let n = tokens.len();
        let mut batch = LlamaBatch::with_capacity(n, 1);
        for (i, &tok) in tokens.iter().enumerate() {
            batch.add(tok, i as sys::llama_pos, 0, true);
        }
        // SAFETY: ctx is live; batch arrays live until `batch` drops after this returns.
        let rc = unsafe { sys::llama_decode(self.ptr.as_ptr(), batch.raw) };
        if rc < 0 {
            return Err(LlamaCppError::Decode(rc));
        }
        let n_out = self.model.n_embd_out() as usize;
        let vocab = self.model.vocab();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: vocab is live (borrowed model); is_control classifies the i-th token.
            let is_ctrl = unsafe { sys::llama_vocab_is_control(vocab, tokens[i]) };
            if is_ctrl {
                continue; // special token: present in the forward pass, excluded from scoring
            }
            // SAFETY: ctx live; ith token's row under pooling=none, n_out contiguous f32.
            let eptr = unsafe { sys::llama_get_embeddings_ith(self.ptr.as_ptr(), i as i32) };
            if eptr.is_null() {
                return Err(LlamaCppError::NoEmbeddings);
            }
            let mut row = unsafe { std::slice::from_raw_parts(eptr, n_out) }.to_vec();
            l2_normalize(&mut row); // per-token L2 → dot product == cosine in MaxSim
            rows.push(row);
        }
        if rows.is_empty() {
            return Err(LlamaCppError::EmptyInput);
        }
        Ok(rows)
    }

    fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<sys::llama_token>> {
        let vocab = self.model.vocab();
        let bytes = text.as_bytes();
        let text_ptr = bytes.as_ptr() as *const c_char;
        let text_len = bytes.len() as i32;
        let mut cap = text_len + 8;
        let mut toks = vec![0 as sys::llama_token; cap.max(1) as usize];
        // SAFETY: vocab is live (from the borrowed model); text_ptr/len describe a valid slice;
        // toks has cap capacity. Returns n on success or -(needed) if the buffer is too small.
        let n = unsafe {
            sys::llama_tokenize(vocab, text_ptr, text_len, toks.as_mut_ptr(), cap, add_special, false)
        };
        let n = if n < 0 {
            cap = -n;
            toks = vec![0 as sys::llama_token; cap as usize];
            // SAFETY: same contract, now with a sufficiently large buffer.
            let n2 = unsafe {
                sys::llama_tokenize(vocab, text_ptr, text_len, toks.as_mut_ptr(), cap, add_special, false)
            };
            if n2 < 0 {
                return Err(LlamaCppError::Tokenize(n2));
            }
            n2
        } else {
            n
        };
        toks.truncate(n as usize);
        Ok(toks)
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        // SAFETY: freed exactly once; the borrow of `model` guarantees the model still lives.
        unsafe { sys::llama_free(self.ptr.as_ptr()) };
    }
}

/// RAII wrapper over `llama_batch_init`/`llama_batch_free` with a manual `add` (the public C API has
/// no `llama_batch_add` — that lives in `common/`). Used for the per-token ColBERT path, where every
/// position must be flagged `logits = 1` so `llama_get_embeddings_ith` returns its row.
struct LlamaBatch {
    raw: sys::llama_batch,
}

impl LlamaBatch {
    /// Allocate a token-mode batch (`embd = 0`) sized for `n_tokens`, one sequence. `n_tokens` starts
    /// at 0; fill with [`add`](Self::add).
    fn with_capacity(n_tokens: usize, n_seq_max: i32) -> Self {
        // SAFETY: standard allocation; leaves all arrays uninitialized (we fill them in `add`).
        let mut raw = unsafe { sys::llama_batch_init(n_tokens as i32, 0, n_seq_max) };
        raw.n_tokens = 0;
        LlamaBatch { raw }
    }

    /// Append one token at `pos` in sequence `seq`, flagging whether its output (logits/embeddings)
    /// is requested. Caller must not exceed the `n_tokens` capacity passed to [`with_capacity`].
    fn add(&mut self, token: sys::llama_token, pos: sys::llama_pos, seq: sys::llama_seq_id, logits: bool) {
        let i = self.raw.n_tokens as usize;
        // SAFETY: i < capacity (caller contract); all arrays were allocated by llama_batch_init for
        // `capacity` entries, and seq_id[i] holds `n_seq_max >= 1` slots.
        unsafe {
            *self.raw.token.add(i) = token;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = 1;
            *(*self.raw.seq_id.add(i)) = seq;
            *self.raw.logits.add(i) = logits as i8;
        }
        self.raw.n_tokens += 1;
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        // SAFETY: raw came from llama_batch_init and is freed exactly once here.
        unsafe { sys::llama_batch_free(self.raw) };
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Matryoshka reduction: truncate `v` to the first `dim` components (a prefix slice — the Matryoshka
/// head, no projection) and **then** L2-normalize the slice. `dim == None` keeps the full vector.
///
/// Order is load-bearing and matches the ATOM reference exactly (`model_runner.py:1817-1819`):
/// slice FIRST, normalize SECOND. The returned vector has unit L2 norm at the requested
/// dimensionality. `dim` larger than `v.len()` is clamped (returns the whole vector).
pub fn matryoshka_truncate_normalize(mut v: Vec<f32>, dim: Option<usize>) -> Vec<f32> {
    if let Some(d) = dim {
        v.truncate(d.min(v.len()));
    }
    l2_normalize(&mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Matryoshka order is load-bearing: TRUNCATE first, THEN L2-normalize. This is a pure
    /// host-math test (no GPU/model needed). raw=[3,4,12] (‖·‖=13); truncated to dim 2 → [3,4], then
    /// normalized over those 2 dims → [0.6, 0.8] (unit norm at dim 2). The WRONG order
    /// (normalize-full-then-slice) would give [3/13, 4/13] = [0.2308, 0.3077] — a non-unit slice.
    #[test]
    fn matryoshka_slices_before_normalizing() {
        let out = matryoshka_truncate_normalize(vec![3.0, 4.0, 12.0], Some(2));
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.6).abs() < 1e-6, "got {:?}", out);
        assert!((out[1] - 0.8).abs() < 1e-6, "got {:?}", out);
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "truncated slice must be unit L2, got {norm}");

        // dim=None → full vector, unit norm (identical to embed()).
        let full = matryoshka_truncate_normalize(vec![3.0, 4.0, 12.0], None);
        assert_eq!(full.len(), 3);
        let fnorm: f32 = full.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((fnorm - 1.0).abs() < 1e-6);
        // dim larger than the vector is clamped, not an error.
        assert_eq!(matryoshka_truncate_normalize(vec![1.0, 0.0], Some(99)).len(), 2);
    }

    /// Per-token ColBERT multi-vector on the gfx1030 HIP base (LFM2-ColBERT-350M f16): one row per
    /// token, each of width `n_embd_out` (=128, the in-graph dense head under pooling=none).
    #[test]
    fn colbert_per_token_on_gpu() {
        let path = "/home/local/ai/models/registry/LiquidAI/LFM2-ColBERT-350M/lfm2-colbert-350m-f16.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: model not found at {path}");
            return;
        }
        let backend = LlamaBackend::init();
        let model = LlamaModel::load(&backend, path, 999).expect("load");
        let n_out = model.n_embd_out();
        let mut ctx = model.embedding_context(512, Pooling::None).expect("ctx");
        let rows = ctx.embed_tokens("[D] def add(a, b): return a + b").expect("embed_tokens");
        eprintln!("n_embd_out={n_out} n_rows={} row0_len={}", rows.len(), rows.first().map_or(0, |r| r.len()));
        assert_eq!(n_out, 128, "LFM2-ColBERT dense head is 128-dim");
        assert!(rows.len() >= 3, "expected several token rows, got {}", rows.len());
        assert!(rows.iter().all(|r| r.len() == n_out as usize), "every row is n_embd_out wide");
        assert_ne!(rows[0], rows[1], "distinct tokens should give distinct rows (not degenerate)");
        // rows are RAW (unnormalized); their norms should generally differ from 1.0.
        let n0: f32 = rows[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("row0 raw L2 norm = {n0:.4}");
    }

    /// ColBERT token prep on the gfx1030 HIP base: per-token rows for `[D] ...`, control tokens
    /// dropped from scoring, each kept row L2-normalized (so dot == cosine for MaxSim).
    #[test]
    fn colbert_tokens_masked_on_gpu() {
        let path = "/home/local/ai/models/registry/LiquidAI/LFM2-ColBERT-350M/lfm2-colbert-350m-f16.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: model not found at {path}");
            return;
        }
        let backend = LlamaBackend::init();
        let model = LlamaModel::load(&backend, path, 999).expect("load");
        let mut ctx = model.embedding_context(512, Pooling::None).expect("ctx");
        let rows = ctx.colbert_tokens("[D] def add(a, b): return a + b", Some(512)).expect("colbert_tokens");
        eprintln!("kept rows={} width={}", rows.len(), rows.first().map_or(0, |r| r.len()));
        assert!(rows.len() >= 3);
        // every kept row is unit-norm (per-token L2 for MaxSim cosine).
        for r in &rows {
            assert_eq!(r.len(), 128);
            let n: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "row not unit-norm: {n}");
        }
    }

    /// End-to-end embedding on the gfx1030 HIP base (qwen3-embedding-0.6B Q8_0).
    #[test]
    fn embed_qwen3_06b_on_gpu() {
        let path =
            "/home/local/ai/models/registry/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: model not found at {path}");
            return;
        }
        let backend = LlamaBackend::init();
        let model = LlamaModel::load(&backend, path, 999).expect("load");
        let n_embd = model.n_embd();
        let mut ctx = model.embedding_context(512, Pooling::Last).expect("ctx");
        let v = ctx.embed("def add(a, b): return a + b").expect("embed");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("n_embd={n_embd} dim={} norm={norm:.5} first3={:?}", v.len(), &v[..3.min(v.len())]);
        assert_eq!(v.len(), n_embd as usize);
        assert!((norm - 1.0).abs() < 1e-3, "expected L2-normalized embedding");
    }
}
