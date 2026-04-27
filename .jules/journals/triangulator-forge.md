 2026-04-25 - Architectural Pattern Identification: The Orchestrator vs The Engine
Learning: There is a recurring pattern in LLM serving where highly popular repositories (like Ollama) are essentially orchestration shells around native execution engines (like llama.cpp). These shells provide superior DX but limited novel execution architecture, while "Engine" repositories (vLLM, TGI, SGLang) entangle execution with heavy ML frameworks (PyTorch, Ray).
Action: Focus future audits on identifying the precise boundary between the "Orchestrator" and the "Engine" to avoid importing framework entanglement when only API ergonomics are desired.

2026-04-26 - SGLang RadixAttention Extraction Feasibility
Learning: SGLang achieves its speed largely through Python-bound orchestration of custom kernels (`sglang/srt/managers`), heavily entangling its advanced RadixAttention caching mechanism with PyTorch. Triton Server similarly buries its core batching logic under layers of gRPC and enterprise C++ abstraction.
Action: For high-performance C++ inference engines like llama.cpp, avoid importing complex orchestration logic or framework bindings. Instead, extract pure conceptual algorithms (e.g., Radix Tree KV slot mapping) and implement them natively in C/C++ without pulling in external ecosystem baggage.

 2026-04-25 - C++ Abstraction Penalty
Learning: Enterprise standard C++ serving engines (like Triton Inference Server) suffer from an abstraction penalty. Their deep, heavily templated, virtual interface-driven architectures provide extreme multi-backend flexibility at the cost of massive compile times, brittle build chains, and poor developer experience for core logic tracing.
Action: When extracting features from enterprise C++ systems, prioritize algorithmic translation over structural porting. We should steal the dynamic batching heuristics, not the abstract class hierarchies.


2026-04-27 - C++ Orchestrator API Boundary Isolation
Learning: Ollama demonstrates that a clean API routing boundary (Go/HTTP) completely separated from execution engine logic prevents orchestrator responsibilities from leaking into the tensor execution layer, resulting in superior DX.
Action: For our engine, ensure any HTTP or external interface routing strictly adheres to the 'Ollama pattern' of abstracting the backend engine behind a narrow API boundary, and avoid the vLLM pitfall of heavily coupling web frameworks (FastAPI) and orchestration (Ray) with tensor execution logic.
