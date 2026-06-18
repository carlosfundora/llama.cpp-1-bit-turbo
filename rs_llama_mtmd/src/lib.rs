//! Multimodal context orchestrator bridging rs_mtmd_utils and llama.cpp's mtmd_context.

// use rs_mtmd_utils::{MtmdContextParams, MtmdBitmap};
// use rs_llama_cpp_core::LlamaContext;

pub struct MtmdContext {
    pub raw_ptr: *mut std::ffi::c_void,
}

impl MtmdContext {
    pub fn new(raw_ptr: *mut std::ffi::c_void) -> Self {
        Self { raw_ptr }
    }

    pub fn decode_use_mrope(&self) -> bool {
        // Intercepts Multimodal Rotary Position Embedding state
        true 
    }

    pub fn decode_use_non_causal(&self) -> bool {
        // Intercepts whether non-causal attention routing is required
        true
    }
}
