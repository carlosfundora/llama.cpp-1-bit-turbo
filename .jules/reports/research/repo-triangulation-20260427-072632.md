# Comparative Repository Audit: vLLM vs Ollama vs SGLang

## 1. Executive Technical Summary
This audit triangulates three dominant but architecturally divergent LLM serving paradigms: **vLLM** (the academic engine and heavy framework), **Ollama** (the Go-based orchestrator wrapping C++ cores), and **SGLang** (the high-performance, radix-attention specialized scheduler).
- **vLLM** provides maximal backend compatibility at the cost of extreme PyTorch/Ray entanglement and monolithic Python orchestration, making it a "heavy framework shell".
- **Ollama** demonstrates a clean Go API/CLI boundary that dynamically orchestrates native C++ engines (predominantly `llama.cpp`), scoring perfectly on developer experience (DX) and integration readiness, but offering little novel execution logic itself.
- **SGLang** possesses the strongest "engine-level" innovation via its RadixAttention state management and customized Triton/CUDA kernels, though it risks severe PyTorch lock-in and deep nesting complexity.

For our project, **Ollama** serves as the optimal reference architecture for API orchestration and system boundaries, while **SGLang** provides the best conceptual algorithms (e.g., prefix caching) for native C++ extraction.

## 2. Repository Targets & Assumptions
- **Repo A**: `vLLM` (vllm-project/vllm) - Assumed to represent the PyTorch/Ray monolithic standard for high-throughput serving.
- **Repo B**: `Ollama` (ollama/ollama) - Assumed to represent the Go-based, DX-first, wrapper/orchestrator paradigm.
- **Repo C**: `SGLang` (sgl-project/sglang) - Assumed to represent the specialized, high-performance structured generation and aggressive caching engine.

## 3. Per-Repo Deep Audit

### 3.1 Repo A: vLLM
- **Core Architecture & Logic Flow**: A heavy monolithic application layer built tightly on PyTorch, Ray (for distributed execution), and FastAPI. The entrypoint flows through an `AsyncLLMEngine` to an `EngineCore` which dispatches to a heavily templated, nested `Scheduler`. The hot path execution engine relies on a mix of PyTorch built-ins, Triton kernels, and custom C++ (`vllm/csrc`) extensions, creating a massive framework shell.
- **Functional Decomposition & "The Heart"**: The heart lives in `vllm/engine/llm_engine.py` and `vllm/core/scheduler.py` (and their v1 equivalents). Logic is smeared heavily across Python orchestration and deeply nested C++ bindings. Complexity Score: **8/10** (due to heavy Python/C++ boundary jumping and Ray entanglement).
- **Dependency & Health Audit**: Dependency-heavy. `requirements/common.txt` lists 40+ deep dependencies including `torch`, `ray`, `transformers`, `xformers`, and `triton`. High transitive fragility. Pulse is very active but maintainer concentration leans towards academic/corporate sponsors.
- **DX & Integration**: Boilerplate for custom integration is high unless using the standard HTTP server. Internal APIs are highly stateful and over-abstracted. Migration risk is **High** due to lock-in to its specific scheduling and KV cache abstractions.

### 3.2 Repo B: Ollama
- **Core Architecture & Logic Flow**: A Go-based modular monolith orchestrator. Entrypoints in `cmd/cmd.go` route to `server/routes.go`. The actual engine is abstracted. The orchestrator prepares the model and dynamically links/calls into CGO or wrapped C++ binaries (`llama.cpp`).
- **Functional Decomposition & "The Heart"**: The heart is not an execution engine but a router/orchestrator, primarily in `llm/server.go` and `server/routes.go`. The hot path is cleanly isolated behind an API boundary. Complexity Score: **3/10** (highly modular, clean Go channels, clear separation of concerns).
- **Dependency & Health Audit**: Flat, controlled Go dependencies (`go.mod`), primarily focused on CLI/API (Gin, Cobra) rather than heavy ML frameworks. Zero PyTorch entanglement. Very healthy, rapid release cadence.
- **DX & Integration**: Setup friction is near zero. The integration surface (REST API, simple Go interfaces) is highly stable and wrapper-friendly. Migration risk is **Low**.

### 3.3 Repo C: SGLang
- **Core Architecture & Logic Flow**: A hybrid architecture focusing on a Python-based scheduler (`sglang/srt/managers/scheduler.py`) and a custom kernel library (`sgl-kernel`). It orchestrates via ZeroMQ/PyTorch Distributed but relies on specialized CUDA/Triton implementations for its RadixAttention mechanism.
- **Functional Decomposition & "The Heart"**: The true innovation lies in `sglang/srt/managers/scheduler.py` and the `sgl-kernel/csrc/` directory where prefix tree logic and flash infer operations occur. However, the orchestration is deeply entangled with PyTorch (`import torch` everywhere in the scheduler). Complexity Score: **7/10** (cleaner than vLLM, but still framework-bound).
- **Dependency & Health Audit**: Requires PyTorch, Triton, and specific CUDA versions. Dependency tree is deep but focused purely on performance execution.
- **DX & Integration**: Harder to integrate purely as a library compared to Ollama, but conceptually rich. Migration risk is **Moderate to High** depending on how deeply the RadixTree state is coupled to the consumer.

## 4. Feature Parity Table
| Feature | vLLM | Ollama | SGLang |
|---|---|---|---|
| Native C++ Engine | Partial (C++ extensions) | No (Orchestrator over `llama.cpp`) | Partial (`sgl-kernel`) |
| Radix/Prefix Caching | Yes | No | **Yes (Best-in-class)** |
| Framework Free | No (PyTorch/Ray) | **Yes (Go/CGO)** | No (PyTorch/ZMQ) |
| Multi-node Distributed | Yes (Ray) | No | Yes |
| Clean API Orchestration| Moderate (FastAPI) | **Excellent (Go/Gin)** | Moderate |

## 5. Comparative Trade-off Matrix
| Metric | vLLM | Ollama | SGLang |
|---|---|---|---|
| **Architectural Clarity** | 5 (Smeared logic) | **9** (Clean boundary) | 7 (Focused scheduler) |
| **Maintainability** | 4 (Brittle deps) | **9** (Go core) | 6 (Custom kernels) |
| **Extensibility** | 8 (Plugins exist) | 7 (Backend wrapper) | 7 (Kernel focused) |
| **Performance Potential** | 8 (High throughput) | 6 (Dependent on backend) | **10** (Optimal routing) |
| **Dependency Risk** | 2 (Severe hell) | **9** (Very lean) | 4 (Torch bound) |
| **Migration Flexibility**| 3 (High lock-in) | **9** (API level) | 5 (State lock-in) |
| **DX / Onboarding** | 6 (Heavy setup) | **10** (One binary) | 6 (Complex build) |
| **Test Trustworthiness** | 5 (Brittle) | **8** (Stable) | 6 (Complex) |
| **Operational Maturity** | **9** (Battle tested) | 8 (Maturing fast) | 6 (Experimental) |
| **Integration Readiness** | 5 (Invasive) | **10** (API Wrapper) | 4 (Tight coupling) |
| **Licensing Suitability** | 10 (Apache 2.0) | 10 (MIT) | 10 (Apache 2.0) |

*(Note: Scores 1-10. Higher is better/safer).*
- *vLLM scores low on Test Trustworthiness due to heavy ML framework fixture dependencies, but high on Operational Maturity given its enterprise adoption.*
- *Ollama scores perfectly on DX and Integration Readiness because of its zero-config single-binary philosophy, and its MIT license ensures maximum flexibility.*
- *SGLang scores top for performance potential due to its native RadixAttention kernels, but lower on Operational Maturity as it is still highly experimental.*
- *vLLM scores low on maintainability due to heavy framework entanglement.*
- *Ollama scores perfectly on DX because of its zero-config single-binary philosophy.*
- *SGLang scores top for performance potential due to its native RadixAttention kernels.*

## 6. Integration Opportunity Mapping
**Opportunity 1: Prefix Caching Algorithms (from SGLang)**
- **Where**: `sglang/srt/managers/scheduler.py` and RadixTree implementations.
- **Why**: SGLang's method of state-sharing via RadixAttention is superior for multi-turn chat and agentic workflows.
- **Difficulty**: High.
- **Recommendation**: **Use as inspiration only.** Do not import the Python orchestration. Extract the Radix algorithm conceptually and implement natively in our C++ core, avoiding PyTorch entanglement. This is a **Strategic Extraction**.

**Opportunity 2: Clean API Orchestration Boundary (from Ollama)**
- **Where**: `server/routes.go` and the `llm/` backend interface.
- **Why**: Ollama cleanly isolates the HTTP/API routing from the execution engine, allowing the engine to be swapped or updated without affecting consumer DX.
- **Difficulty**: Low.
- **Recommendation**: **Adopt architectural pattern.** If our project needs a server wrapper, we should emulate this strict boundary layer. This is a **Fast Win**.

**Opportunity 3: Avoid: Deep PyTorch Scheduler Binding (from vLLM)**
- **Where**: `vllm/v1/engine/core.py`.
- **Why**: vLLM buries execution inside heavy Python abstractions that require Ray for scaling, leading to the "C++ Abstraction Penalty" noted in previous journals.
- **Recommendation**: **Avoid importing.** This is an **Attractive Trap**.

## 7. Adoption Plan
**Target Architecture**: Maintain our native C++ execution core (`llama.cpp` style) while building a lightweight, non-framework orchestration API modeled after Ollama. We will conceptually port SGLang's RadixAttention caching logic directly into our native KV cache layer without adopting any of SGLang's Python/Triton dependencies.

- **Phase 1 (Seams)**: Establish a clean API/Engine boundary (Ollama pattern).
- **Phase 2 (Isolation)**: Sandbox our KV Cache module.
- **Phase 3 (Rollout)**: Introduce native C++ Radix Tree mapping into the sandboxed KV Cache.
- **Fallback Strategy**: If native C++ Radix tree extraction fails or degrades performance, revert to standard KV slot allocation and use Ollama solely as a black-box container orchestrator for our existing engine.
- **Wrap rather than fork**: Wrap Ollama's `server/routes.go` logic via our own HTTP interface instead of forking their entire Go repository.
- **Fork rather than wrap**: N/A. We will avoid forking. We will *steal* the RadixTree conceptual algorithm from SGLang but not fork the codebase.

## 8. Concrete Work Items
1. **Ticket: Implement Native C++ Radix Tree KV Cache Slot Mapping**
   - **Purpose**: Conceptually port SGLang's RadixAttention caching.
   - **Area**: `common/hf-cache` or `ggml` KV cache layers.
   - **Dependency Order**: 2
   - **Risk**: High
   - **Acceptance**: Cache hits significantly improve for overlapping prompt prefixes.
2. **Ticket: Define Strict Engine API Boundary (The Ollama Pattern)**
   - **Purpose**: Prevent internal execution logic from leaking into HTTP handlers.
   - **Area**: `server/` module.
   - **Dependency Order**: 1
   - **Risk**: Low
   - **Acceptance**: Server routes no longer directly manipulate tensor state.
3. **Ticket: Remove Framework Leakage from Hot Paths**
   - **Purpose**: Ensure no heavy abstraction penalty.
   - **Area**: Core scheduling loop.
   - **Dependency Order**: 3
   - **Risk**: Medium
   - **Acceptance**: Hot path benchmark shows <5% orchestration overhead.

**Suggested First PR**: Define the strict Engine API boundary (Ticket 2) to create a safe foundation.
**Suggested Second PR**: Extract the Radix Tree logic conceptually into a standalone C++ data structure (Ticket 1).

**What NOT to do**: Do not add PyTorch or Ray as dependencies for scheduling or KV cache management.

## 9. Final Recommendation
- **Best For Production Stability & Developer Ergonomics**: `Ollama`
- **Best For Experimental High-Upside Performance**: `SGLang`
- **Best To Learn From, Not Adopt**: `vLLM` (due to heavy framework entanglement)

**Verdict**:
- **Best to Adopt Directly**: `Ollama` (as a pattern for API orchestration).
- **Best to Fork/Adapt**: `SGLang` (specifically its RadixAttention logic, ported conceptually to our native C++).
- **Best Avoided**: `vLLM` (without a heavy refactor, its PyTorch-bound orchestration creates an unacceptable abstraction penalty for a lean native engine).

## 10. Horizon Scanning
1. **The Rising Star**: `TGI (Text Generation Inference)` by HuggingFace. Fills the Rust-based orchestration niche. Important because it proves Rust can provide memory-safe, high-concurrency orchestration without PyTorch's Python GIL overhead. Skipped here as it overlaps architecturally with vLLM's monolith approach but in Rust.
2. **The Legacy Standard**: `Triton Inference Server`. The enterprise C++ standard. Important for its multi-backend flexibility. Skipped because previous audits proved it suffers from a massive "C++ Abstraction Penalty" making it too heavy for lean deployments.
3. **The Niche Specialist**: `ExLlamaV2`. Fills the extreme low-VRAM, quantization-first local serving niche. Important for its aggressively optimized custom kernels for consumer GPUs. Skipped because it focuses narrowly on specific quant formats rather than general orchestration.

## 11. Appendix: Evidence Notes
- vLLM heavily relies on PyTorch/Ray as seen in `vllm/v1/engine/core.py` and `requirements/common.txt`.
- Ollama is primarily Go (`go.mod` has 0 ML framework deps) and orchestrates via `cmd/` and `server/routes.go`.
- SGLang contains advanced custom kernels (`sgl-kernel/csrc/`) but deeply embeds `import torch` throughout its `sglang/srt/managers/scheduler.py` hot path.
