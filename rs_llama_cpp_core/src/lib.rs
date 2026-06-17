//! Safe wrapper around `llama_context` and `llama_batch` with strict lifetime and initialization checks.

use std::marker::PhantomData;

/// Represents a raw pointer to a llama_context
#[derive(Debug)]
pub struct LlamaContextRaw(*mut std::ffi::c_void);

/// Safe wrapper around a LlamaContext
pub struct LlamaContext<'a> {
    pub raw: LlamaContextRaw,
    pub initialized_logits: Vec<i32>,
    pub embeddings_enabled: bool,
    _phantom: PhantomData<&'a ()>, // Binds the context lifetime
}

impl<'a> LlamaContext<'a> {
    pub fn new(raw: *mut std::ffi::c_void, embeddings_enabled: bool) -> Self {
        Self {
            raw: LlamaContextRaw(raw),
            initialized_logits: Vec::new(),
            embeddings_enabled,
            _phantom: PhantomData,
        }
    }

    /// Verifies logits are initialized before allowing access
    pub fn check_logits_initialized(&self, index: i32) -> Result<(), String> {
        if self.initialized_logits.contains(&index) {
            Ok(())
        } else {
            Err(format!("Logits for index {} are not initialized", index))
        }
    }
}

pub struct LlamaBatch<'a> {
    pub allocated: usize,
    pub n_tokens: usize,
    pub initialized_logits: Vec<i32>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> LlamaBatch<'a> {
    pub fn new(allocated: usize) -> Self {
        Self {
            allocated,
            n_tokens: 0,
            initialized_logits: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn add(&mut self, _token: i32, _pos: i32, logits: bool) -> Result<(), String> {
        if self.n_tokens >= self.allocated {
            return Err("Insufficient Space".to_string());
        }
        
        let offset = self.n_tokens as i32;
        if logits {
            self.initialized_logits.push(offset);
        }
        self.n_tokens += 1;
        Ok(())
    }
}
