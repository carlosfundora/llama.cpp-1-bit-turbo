# Comparative Repository Audit: vLLM vs SGLang vs Ollama

## 1. Executive Technical Summary
This audit comparatively evaluates **vLLM**, **SGLang**, and **Ollama** to identify the most robust architectural patterns and extraction opportunities for our internal inference engine project (`llama.cpp`-based).

We found a sharp divergence between **"Orchestrator"** and **"Engine"** architectures. **Ollama** serves as a highly modular, DX-first Go orchestrator that explicitly decouples its REST/routing logic from actual model execution. **vLLM** and **SGLang** are tightly coupled Python engines that deeply intertwine high-performance serving concepts (continuous batching, prefix caching) with heavy ML frameworks (PyTorch, Ray, FastAPI).

The primary conclusion is that while **vLLM** and **SGLang** possess the most advanced algorithmic innovations (e.g., RadixAttention), their codebases suffer from severe framework entanglement and lock-in risk. We should **selectively extract algorithms** (like Radix KV management) rather than adopting their architectures. Conversely, **Ollama's** clean Go boundary represents a **Best Reference Architecture** for building our API and orchestration layers around our C++ core.

## 2. Repository Targets & Assumptions
- **Repo A:** `vLLM` (vllm-project/vllm) - Inferred as the leading high-throughput serving engine standard.
- **Repo B:** `SGLang` (sgl-project/sglang) - Inferred as the cutting-edge structural/prefix-caching engine.
- **Repo C:** `Ollama` (ollama/ollama) - Inferred as the dominant developer-experience and orchestration wrapper.
Assumed that `llama.cpp` is our native execution engine, and we are evaluating these repos for architectural extraction opportunities.

## 3. Per-Repo Deep Audit

### 3.1 Repo A: vLLM
#### Core Architecture & Logic Flow
vLLM is a monolithic, framework-bound application centered around PyTorch and Ray. It acts as the actual execution engine. The primary data flow routes REST requests via FastAPI directly into a Python event loop that orchestrates asynchronous generation.
- **Concurrency:** Async/await (asyncio) tied to Ray for distributed execution.
- **Shape:** Framework-bound application / monolithic engine.

#### Functional Decomposition & "The Heart"
- **Hot Path:** `vllm/engine/async_llm_engine.py` into `vllm/worker/worker.py` and ultimately custom CUDA kernels (`vllm/vllm_flash_attn`, `vllm/attention/ops`).
- **Isolation:** Poor. The hot path is smeared across PyTorch abstractions and custom memory managers (`vllm/core/kv_cache_manager.py`).
- **Complexity Score:** 8/10. Deeply nested, heavily coupled to PyTorch versions, and relies extensively on complex C++ extensions mapped to Python.
- **Feature Parity:** FastAPI surface, paged attention, multi-node Ray.
- **Unique Value:** Industry-standard PagedAttention implementation.

#### Dependency & Health Audit
- **Dependencies:** Deep and fragile. Heavy reliance on specific PyTorch builds, CUDA versions, xformers, and triton (`requirements/common.txt` and `setup.py`).
- **Health:** Rapidly iterating, massive contributor base, but prone to dependency hell (as seen in complex wheel building logic in `setup.py`).
- **License:** Apache 2.0.

#### Developer Experience & Integration
- **DX:** Requires significant boilerplate to extract its core logic without pulling in FastAPI/Ray.
- **Tests:** End-to-end tests exist, but mock/unit tests are heavily coupled to PyTorch fixtures. Tests protect the hot path but are slow.
- **Integration:** Invasive and wrapper-friendly only if using Python. Clean integration into a C++ engine is impossible without a complete rewrite.

#### Lock-in & Migration Risk
- **Risk:** Severe.
- **Why:** Adopting vLLM means adopting PyTorch and Python as the execution environment, which fundamentally breaks our lightweight C++ strategy. It must be avoided for direct adoption.

### 3.2 Repo B: SGLang
#### Core Architecture & Logic Flow
SGLang is a customized execution engine built for structural generation and prefix caching. Like vLLM, it is bound to PyTorch but focuses on a RadixAttention cache.
- **Concurrency:** Multi-processing and threading orchestrating custom asynchronous schedulers (`sglang/srt/managers/scheduler.py`).
- **Shape:** Plugin-heavy framework shell wrapping custom execution logic.

#### Functional Decomposition & "The Heart"
- **Hot Path:** The true heart is the Radix tree implementation for prefix caching (`python/sglang/srt/mem_cache/radix_cache.py`) managed by Python schedulers.
- **Isolation:** Better than vLLM conceptually, but practically tied to PyTorch tensors for KV pool management.
- **Complexity Score:** 7/10. Abstractions are clearer than vLLM, particularly around memory caching (`sglang/srt/mem_cache/`), but still heavily Python-bound.
- **Feature Parity:** FastAPI surface, prefix caching, grammar constraints.
- **Unique Value:** RadixAttention KV Cache implementation.

#### Dependency & Health Audit
- **Dependencies:** Framework-heavy (PyTorch, triton, ray) managed via `python/pyproject.toml`.
- **Health:** Fast-moving, actively stripping away legacy abstractions to optimize throughput.
- **License:** Apache 2.0.

#### Developer Experience & Integration
- **DX:** High friction for non-Python apps.
- **Tests:** Heavily reliant on integration testing against PyTorch.
- **Integration:** Invasive. Porting RadixAttention requires complete translation of their Python data structures to C++.

#### Lock-in & Migration Risk
- **Risk:** High.
- **Why:** The core prefix caching logic is deeply tied to their specific scheduler and memory pool abstractions (`memory_pool.py`). We should only extract the conceptual Radix tree logic.

### 3.3 Repo C: Ollama
#### Core Architecture & Logic Flow
Ollama is a Go-based orchestration layer that wraps native C/C++ backends (primarily `llama.cpp` and `MLX`).
- **Concurrency:** Go routines.
- **Shape:** CLI-first with a service-oriented orchestration core and adapter boundaries.

#### Functional Decomposition & "The Heart"
- **Hot Path:** HTTP handler (`server/routes.go`) -> Scheduler (`server/sched.go`) -> Runner interface (`runner/runner.go`) -> Native C/C++ bindings (`runner/llamarunner/runner.go`).
- **Isolation:** Excellent. The Go API layer knows almost nothing about the tensor execution logic.
- **Complexity Score:** 4/10. Highly modular, clean adapter boundaries, and explicitly designed to prevent execution logic from bleeding into orchestration.
- **Feature Parity:** Gin REST API, model persistence (SQLite/fs), runner abstraction.
- **Unique Value:** Clean orchestrator-engine separation boundary.

#### Dependency & Health Audit
- **Dependencies:** Lean and controlled (`go.mod`), relying mostly on standard web/CLI tools (gin, cobra) and CGO for native binding.
- **Health:** Extremely stable, focused on DX and portability.
- **License:** MIT.

#### Developer Experience & Integration
- **DX:** Excellent. Very low boilerplate to add new endpoints.
- **Tests:** Solid unit tests for the scheduler and API handlers.
- **Integration:** Clean and wrapper-friendly. The runner interface (`Runner`) is a perfect model for an orchestrator.

#### Lock-in & Migration Risk
- **Risk:** Low.
- **Why:** Because execution is delegated to isolated runners, replacing the underlying engine (e.g., swapping llama.cpp for MLX) is trivial.

## 4. Feature Parity Table

| Feature | Repo A: vLLM | Repo B: SGLang | Repo C: Ollama |
| :--- | :--- | :--- | :--- |
| **API Surface** | FastAPI REST | FastAPI REST | Gin REST |
| **Persistence/State** | In-memory | In-memory | SQLite / FS |
| **KV Caching Strategy** | PagedAttention | RadixAttention | Engine-delegated |
| **Execution Engine** | PyTorch / Custom CUDA | PyTorch / Triton | llama.cpp / MLX |
| **Multi-Node** | Ray | Ray / Custom | None |
| **Adapter Boundary** | Poor | Poor | Excellent |

## 5. Comparative Trade-off Matrix

| Metric | Repo A: vLLM | Repo B: SGLang | Repo C: Ollama |
| :--- | :--- | :--- | :--- |
| **Architectural Clarity** | 4/10 (Monolith; logic smeared) | 6/10 (Modular Python, tied to PyTorch) | 9/10 (Clean boundaries, clear separation) |
| **Maintainability** | 3/10 (Dependency hell; PyTorch/CUDA limits) | 5/10 (Python bound; fast-moving) | 8/10 (Go simplicity; minimal deps) |
| **Extensibility** | 5/10 (High friction for non-Python) | 7/10 (Better decoupled than vLLM) | 8/10 (Adapter design makes extension easy) |
| **Performance Potential** | 9/10 (Raw throughput via custom kernels) | 10/10 (Prefix matching is highly optimized) | 7/10 (Delegated; overhead from IPC/CGO) |
| **Dependency Risk** | 8/10 (Severe; brittle ML packages) | 7/10 (High; tied to Triton/Ray) | 2/10 (Low; standard Go libs) |
| **Migration Flexibility** | 2/10 (High lock-in to their APIs) | 3/10 (High lock-in to scheduler) | 8/10 (Easy to swap backend runners) |
| **DX / Onboarding** | 4/10 (Complex build system) | 5/10 (Easier but still Python-heavy) | 9/10 (Simple CLI and standard Go) |
| **Test Trustworthiness** | 7/10 (Good E2E, but mock heavy) | 6/10 (Integration tests dominate) | 8/10 (Clear unit tests for routing/sched) |
| **Operational Maturity** | 9/10 (Industry standard production) | 7/10 (Emerging production standard) | 8/10 (Widely deployed consumer/edge) |
| **Integration Readiness** | 2/10 (Invasive to embed in C++) | 3/10 (Invasive for C++ adoption) | 9/10 (Wrapper friendly via API) |
| **Licensing Suitability** | 8/10 (Apache 2.0) | 8/10 (Apache 2.0) | 10/10 (MIT) |

## 6. Integration Opportunity Mapping

1. **RadixAttention KV Caching (SGLang)**
   - **Where:** `python/sglang/srt/mem_cache/radix_cache.py`
   - **Value:** O(log N) prefix matching for prompt sharing across continuous batches.
   - **Difficulty:** High
   - **Recommendation:** Adapt to our architecture natively in C/C++. Do not use Python code.

2. **Clean Runner Scheduling Boundary (Ollama)**
   - **Where:** `server/sched.go` and `runner/runner.go`
   - **Value:** Isolates model loading, VRAM estimation, and orchestration from the execution engine.
   - **Difficulty:** Low
   - **Recommendation:** Use as inspiration only. Adopt the conceptual adapter boundary for our external API layer.

3. **PagedAttention Block Allocation (vLLM)**
   - **Where:** `vllm/core/kv_cache_manager.py`
   - **Value:** Virtual memory-like allocation of KV cache blocks to prevent fragmentation.
   - **Difficulty:** Medium
   - **Recommendation:** Adapt to our architecture. Combine with SGLang's Radix logic.

Fast Wins: Implement the API routing separation (Ollama-style).
Strategic Extractions: Porting SGLang's Radix tree to our C++ KV cache layer.
Attractive Traps: Importing FastAPI or PyTorch serving layers from vLLM.

## 7. Adoption Plan

- **Recommended Target Architecture:** We maintain our pure C/C++ engine (`llama.cpp` core).
- **Seams / Isolation Layers:** Create a strict API/Orchestration boundary (inspired by Ollama) that completely encapsulates the C++ execution engine.
- **Rollout Order:**
  1. Refactor our API server to decouple HTTP handlers from model execution.
  2. Implement an abstract `Runner` interface in our core.
  3. Port RadixAttention logic to C++ behind a `llama_kv_cache` abstraction.
- **Fallback Strategy:** If Radix caching is too complex, fall back to standard PagedAttention block allocation.
- **Wrapping vs Forking:** Avoid importing vLLM/SGLang code entirely. Steal the idea but not the code for Radix caching. Wrap our C++ engine in a clean HTTP boundary inspired by Ollama.

## 8. Concrete Work Items

1. **[Architecture Refactor] Decouple Server Routing from Execution**
   - **Title:** Refactor HTTP Server to Isolate Execution Engine
   - **Purpose:** Emulate Ollama's `server/routes.go` adapter boundary to prevent orchestration logic from bleeding into tensor operations.
   - **Affected Area:** `examples/server/` and core API headers.
   - **Dependency Order:** 1
   - **Risk Level:** Low
   - **Acceptance Criteria:** HTTP handlers have no direct references to tensor operations, only interacting via an abstracted runner/scheduler interface.

2. **[Feature Extraction] C++ Radix Tree KV Slot Mapper**
   - **Title:** Implement Radix Tree for Prefix KV Caching
   - **Purpose:** Extract the pure conceptual algorithm from SGLang's `radix_cache.py` into our C++ KV cache system.
   - **Affected Area:** `common/` or `llama_kv_cache` structures.
   - **Dependency Order:** 2
   - **Risk Level:** High
   - **Acceptance Criteria:** `llama_kv_cache` can perform O(log N) lookup for prompt prefix sharing using a native C++ radix tree, bypassing linear scans.

3. **[Implementation] Continuous Batching Block Allocator**
   - **Title:** Virtual Memory-style KV Block Allocation
   - **Purpose:** Implement PagedAttention-style non-contiguous slot allocation for the KV cache.
   - **Affected Area:** `llama_kv_cache` (specifically `llama_kv_cells`).
   - **Dependency Order:** 3
   - **Risk Level:** Medium
   - **Acceptance Criteria:** The KV cache allocator manages fragmented blocks using O(log N) lookup structures (e.g., `std::set`), avoiding O(N) linear scans during slot assignment.

- **Suggested First PR:** Refactor HTTP Server to Isolate Execution Engine (Creates the safest high-leverage foundation).
- **Suggested Second PR:** Implement Radix Tree for Prefix KV Caching (Extracts one real feature from the research).

### What NOT to do
- Do not introduce PyTorch, Ray, or Python framework bindings into the core engine.
- Do not build custom memory allocators or orchestration layers in Python.
- Do not adopt an enterprise C++ virtual-interface architecture that increases compile times.

## 9. Final Recommendation

- **Ollama** is the **Best Reference Architecture** for system boundaries and DX.
- **SGLang** is the **Best to Mine for Ideas** (specifically its Radix KV caching).
- **vLLM** is the **Best to Learn From, Not Adopt** due to severe framework entanglement and Python lock-in.
- **Verdict:** Do not adopt any of these repositories directly. Mine SGLang for its Radix caching algorithms (porting them to C++) and use Ollama's clean separation of concerns to guide our orchestration layer design.

## 10. Horizon Scanning
1. **The Rising Star:** `Aphrodite-engine`
   - **Why selected:** Fork of vLLM focused on stability and specific edge cases over bleeding-edge features.
   - **Category:** Rapid Prototyping / Inference Fork.
   - **Technical reason:** Demonstrates community frustration with vLLM's rapid, breaking changes and dependency instability.
   - **Why not deep-dived:** Shares the exact same core architecture as vLLM, providing no novel architectural insights.

2. **The Legacy Standard:** `Triton Inference Server`
   - **Why selected:** NVIDIA's enterprise standard for serving.
   - **Category:** Enterprise Integration.
   - **Technical reason:** Highly flexible multi-backend orchestration but suffers from massive C++ abstraction penalties.
   - **Why not deep-dived:** Its deep, templated, virtual interface-driven architecture contradicts our requirement for a lightweight, lean C/C++ engine.

3. **The Niche Specialist:** `TensorRT-LLM`
   - **Why selected:** NVIDIA's proprietary high-performance engine.
   - **Category:** Experimental High-Upside / Hardware Specialist.
   - **Technical reason:** Demonstrates peak possible performance on specific NVIDIA hardware through extreme kernel optimization.
   - **Why not deep-dived:** Severe hardware and ecosystem lock-in makes it completely unsuitable for a portable, cross-platform engine like `llama.cpp`.

## 11. Appendix: Evidence Notes
- vLLM heavily relies on PyTorch C++ extensions (`setup.py` extension mapping logic observed in `vllm-project/vllm`).
- SGLang cleanly isolates its caching logic conceptually (`mem_cache/radix_cache.py`) but practically still manages state via Python schedulers and PyTorch memory pools.
- Ollama explicitly uses `runner` interfaces to isolate backends like `llama.cpp` (`runner/runner.go` and `server/routes.go`), providing excellent architectural decoupling.
