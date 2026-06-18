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
    /// Tokenize + decode + extract the pooled sequence embedding, L2-normalized.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
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
        let mut v = unsafe { std::slice::from_raw_parts(eptr, n_embd) }.to_vec();
        l2_normalize(&mut v);
        Ok(v)
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

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
