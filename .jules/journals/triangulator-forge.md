 2026-04-25 - Architectural Pattern Identification: The Orchestrator vs The Engine
Learning: There is a recurring pattern in LLM serving where highly popular repositories (like Ollama) are essentially orchestration shells around native execution engines (like llama.cpp). These shells provide superior DX but limited novel execution architecture, while "Engine" repositories (vLLM, TGI, SGLang) entangle execution with heavy ML frameworks (PyTorch, Ray).
Action: Focus future audits on identifying the precise boundary between the "Orchestrator" and the "Engine" to avoid importing framework entanglement when only API ergonomics are desired.

 2026-04-25 - C++ Abstraction Penalty
Learning: Enterprise standard C++ serving engines (like Triton Inference Server) suffer from an abstraction penalty. Their deep, heavily templated, virtual interface-driven architectures provide extreme multi-backend flexibility at the cost of massive compile times, brittle build chains, and poor developer experience for core logic tracing.
Action: When extracting features from enterprise C++ systems, prioritize algorithmic translation over structural porting. We should steal the dynamic batching heuristics, not the abstract class hierarchies.
