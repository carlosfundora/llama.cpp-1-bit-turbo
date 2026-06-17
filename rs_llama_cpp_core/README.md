# rs_llama_cpp_core
Safe Rust wrappers around `llama_context`, `llama_model`, and `llama_batch`.
Enforces struct lifetimes using `PhantomData` and tracks initialized logits to prevent unsafe memory access.
