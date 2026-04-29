# Comparative Repository Audit: vLLM vs SGLang vs Ollama

## 1. Executive Technical Summary
This report analyzes three prominent LLM serving frameworks: **vLLM**, **SGLang**, and **Ollama**. The audit focused on their internal architecture, maintainability, dependency risk, and integration viability for our core inference engine (`llama.cpp`).
- **vLLM** provides highly optimized PagedAttention memory management but entangles its execution engine deeply with Ray for distributed orchestration and FastAPI for routing, creating high migration friction.
- **SGLang** introduces a superior RadixAttention cache mechanism (Radix Tree based KV slots), achieving exceptional throughput. However, its hot path is heavily bound to Python orchestration and PyTorch tensors, making direct adoption dangerous for a pure C++ engine.
- **Ollama** shines in Developer Experience (DX). It uses a clean Go/HTTP routing boundary completely separated from its underlying C++ execution engine via the `LlamaServer` interface, presenting a robust pattern for orchestration shells.

**Final Verdict:** We should extract SGLang's RadixAttention algorithmic concepts directly into our C++ `llama_kv_cache` and emulate Ollama's strict orchestration/execution API boundaries, while avoiding the heavyweight framework lock-ins of vLLM and SGLang.

## 2. Repository Targets & Assumptions
- **Repo A:** vLLM (vllm-project/vllm) - Assumed as the industry standard for PagedAttention serving.
- **Repo B:** SGLang (sgl-project/sglang) - Assumed as the cutting-edge implementation of RadixAttention.
- **Repo C:** Ollama (ollama/ollama) - Assumed as the standard for local LLM DX and API orchestration.
All repositories were cloned via `--depth 1` into `/tmp/repos/` for concrete architectural analysis.

## 3. Per-Repo Deep Audit

### Repo A: vLLM
- **Core Architecture & Logic Flow:** Heavy reliance on Python for scheduling. Distributed execution relies heavily on Ray (`vllm/ray/ray_env.py`). Tensor operations and PagedAttention are implemented in C++ (`csrc/rocm/torch_bindings.cpp`) but tightly wrapped with PyTorch bindings.
- **Functional Decomposition & "The Heart":** The hot path involves Python FastAPI routes triggering Ray actors, which call into PyTorch-bound C++ kernels (`PagedAttention.split_kv_cache`). Complexity Score: **7/10** (Deeply nested Ray abstractions).
- **Dependency & Health Audit:** Massive dependency tree (`requirements/common.txt`), relying heavily on `torch`, `ray`, and `fastapi`. Brittle to upgrade but highly active community.
- **DX, Integration, and Risk:** Integration requires adopting Python. Severe framework lock-in risk (Ray/PyTorch).

### Repo B: SGLang
- **Core Architecture & Logic Flow:** Similar to vLLM, it utilizes Python for high-level orchestration but introduces highly optimized custom CUDA/ROCm kernels (`sgl-kernel/csrc/`).
- **Functional Decomposition & "The Heart":** The core innovation is `RadixCache` (`python/sglang/srt/mem_cache/radix_cache.py`), utilizing a `TreeNode` structure where values are `torch.Tensor` objects. Complexity Score: **6/10** (Clean algorithmic idea but smeared across Python/C++ boundary).
- **Dependency & Health Audit:** Dependencies (`python/pyproject.toml`) include `torch`, `triton`, and custom C++ bindings. High maintainer activity.
- **DX, Integration, and Risk:** Integrating its core feature (RadixAttention) directly is difficult due to PyTorch entanglement. Moderate lock-in risk.

### Repo C: Ollama
- **Core Architecture & Logic Flow:** Go-based API shell (`server/routes.go`) that shells out to C++ runners. It uses a strict `llm.LlamaServer` interface (`llm/server.go`) to isolate the engine.
- **Functional Decomposition & "The Heart":** The hot path is a lightweight Gin router parsing HTTP to Go structs, then forwarding via RPC-like calls to the engine. Complexity Score: **3/10** (Very clean, simple wrapper).
- **Dependency & Health Audit:** Lean Go dependencies (`go.mod`), mostly standard library and basic web utilities (`gin-gonic`).
- **DX, Integration, and Risk:** Extreme ease of use. Low lock-in risk since the actual model execution is isolated behind an adapter.

## 4. Feature Parity Table

| Feature | vLLM | SGLang | Ollama |
| :--- | :--- | :--- | :--- |
| **Plugin/Module System** | Yes (LoRA Resolvers) | No | No |
| **Schema Validation** | Yes (FastAPI/Pydantic) | Yes | Yes (Gin) |
| **CLI Support** | Moderate | Moderate | Excellent |
| **Streaming** | Yes | Yes | Yes |
| **Job Orchestration** | Yes (Ray) | Yes (Custom Python) | No (Basic Scheduler) |
| **Caching Mechanism** | PagedAttention | RadixAttention | Basic KV |

## 5. Comparative Trade-off Matrix

| Metric | vLLM | SGLang | Ollama | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Clarity** | 5 | 6 | 9 | Ollama enforces a strict execution boundary; vLLM/SGLang mix orchestration with tensor math. |
| **Maintainability** | 4 | 5 | 8 | Go's static typing and lean deps beat massive PyTorch/Ray stacks. |
| **Extensibility** | 8 | 7 | 4 | vLLM's plugin ecosystem is vast; Ollama is rigidly focused on its core use case. |
| **Performance Potential**| 9 | 10 | 6 | SGLang's Radix cache dominates throughput. Ollama suffers wrapper overhead. |
| **Dependency Risk** | 2 | 3 | 9 | vLLM/SGLang rely heavily on massive ML frameworks. Ollama is lean. |
| **Migration Flexibility**| 2 | 3 | 8 | Disentangling from Ray/PyTorch is extremely hard. Ollama is just an API wrapper. |
| **DX / Onboarding** | 6 | 5 | 10 | Ollama is the gold standard for DX. |
| **Test Trustworthiness** | 7 | 6 | 8 | Ollama's Go tests effectively mock the engine. vLLM has heavy integration suites. |
| **Operational Maturity** | 9 | 7 | 8 | vLLM is enterprise-proven. SGLang is cutting edge. |
| **Integration Readiness**| 4 | 4 | 9 | Ollama's API pattern is ready to wrap any engine. |
| **Licensing Suitability**| 10 | 10 | 10 | All are Apache/MIT licensed. |

## 6. Integration Opportunity Mapping

1. **Radix Tree KV Slot Mapping**
   - **Source:** SGLang (`python/sglang/srt/mem_cache/radix_cache.py`)
   - **Value:** Massive throughput improvements for shared prompt caching.
   - **Difficulty:** High
   - **Recommendation:** Adapt to our architecture. Extract the pure C++ logic of `RadixNode` and implement natively in `llama_kv_cache` without PyTorch.

2. **Clean API Routing Boundary**
   - **Source:** Ollama (`server/routes.go` and `llm/server.go`)
   - **Value:** Prevents orchestration responsibilities from leaking into tensor execution layers.
   - **Difficulty:** Low
   - **Recommendation:** Use as inspiration only. Emulate the `LlamaServer` interface pattern when building our own external API adapters.

3. **PagedAttention Block Management**
   - **Source:** vLLM (`csrc/rocm/torch_bindings.cpp`)
   - **Value:** Efficient non-contiguous KV cache allocation.
   - **Difficulty:** Medium
   - **Recommendation:** Adapt to our architecture. (Already partially present in our TurboMind-style allocators).

**Attractive Trap:** Do not import Ray orchestration or Python-based event loops from vLLM/SGLang into our C++ core.

## 7. Adoption Plan
**Target Architecture:**
We will maintain our pure C++ core engine (`llama.cpp`) but enhance its KV cache subsystem by adapting SGLang's algorithmic Radix tree caching. We will establish a firm isolation layer (similar to Ollama's `LlamaServer`) to ensure API orchestration never bleeds into tensor execution.

**Rollout Order:**
1. Isolate the existing `llama_kv_cache` interface.
2. Implement a pure C++ Radix Tree data structure.
3. Integrate the Radix Tree into `llama_kv_cache` for prefix caching.

## 8. Concrete Work Items

1. **Ticket: C++ Radix Tree Data Structure**
   - **Purpose:** Implement a pure C++ radix tree (`llama_radix_tree.h`) to map token sequences to KV cache slot indices, mirroring the algorithmic behavior of SGLang's `RadixCache`.
   - **Affected Area:** `src/llama-memory.cpp` / `src/llama-memory.h`
   - **Dependency Order:** 1
   - **Risk Level:** Low
   - **Acceptance Criteria:** Unit tests pass demonstrating correct insertion, prefix matching, and eviction via LRU based on sequence hashes.

2. **Ticket: Integrate Radix Cache into llama_kv_cache**
   - **Purpose:** Replace linear prefix matching in KV cache with the new O(log N) Radix Tree lookup.
   - **Affected Area:** `src/llama-memory.cpp`
   - **Dependency Order:** 2
   - **Risk Level:** High
   - **Acceptance Criteria:** `llama_kv_cache_find_slot` successfully utilizes the radix tree. Existing generation tests pass with identical outputs.

3. **Ticket: Formalize C++ API Adapter Boundary**
   - **Purpose:** Create a strict adapter layer (like Ollama's `LlamaServer`) that wraps `llama_context` for external HTTP/RPC orchestration tools, preventing API logic bleed.
   - **Affected Area:** `common/server.cpp`
   - **Dependency Order:** 3
   - **Risk Level:** Medium
   - **Acceptance Criteria:** The server codebase only interacts with the core engine through a defined pure-virtual interface wrapper.

**Suggested First PR:** Implement the pure C++ `llama_radix_tree.h` structure with comprehensive unit tests (Ticket 1). This creates a safe, isolated foundation without modifying core execution paths.
**Suggested Second PR:** Wire the `llama_radix_tree` into the `llama_kv_cache` (Ticket 2).

**What not to do:** Do not add Python bindings inside the core tensor execution loops. Do not implement HTTP routing directly inside `llama_context`.

## 9. Final Recommendation
**Best to Adopt Directly:** Ollama (for its clean architectural boundaries and DX).
**Best to Mine for Ideas:** SGLang (for its RadixAttention logic) and vLLM (for PagedAttention logic).
**Best to Avoid:** Do not attempt to adopt vLLM or SGLang's orchestration engines directly due to severe Python/Ray/PyTorch entanglement.

## 10. Horizon Scanning

1. **The Rising Star: TensorRT-LLM (NVIDIA)**
   - **Category:** Extreme Performance C++ Engine
   - **Reason:** Pushes the absolute limit of Hopper architecture optimizations, providing highly specialized fused kernels. Not deeply audited as it is highly vendor-locked to NVIDIA.
2. **The Legacy Standard: Triton Inference Server**
   - **Category:** Enterprise Orchestration
   - **Reason:** The gold standard for multi-framework (ONNX, PyTorch, TensorRT) model orchestration in Kubernetes. Excluded due to massive C++ abstraction penalties.
3. **The Niche Specialist: MLC-LLM**
   - **Category:** Universal Compilation / Edge
   - **Reason:** Uses TVM compilation to run models on WebGL, WebGPU, and mobile devices. Highly unique but operates on a fundamentally different compilation paradigm (AOT vs JIT/Interpreter) than standard inference engines.

## 11. Appendix: Evidence Notes
- vLLM Python entanglement verified via `vllm/ray/ray_env.py` and `csrc/rocm/torch_bindings.cpp`.
- SGLang RadixCache structure verified via `python/sglang/srt/mem_cache/radix_cache.py`.
- Ollama separation verified via `server/routes.go` and `llm/server.go`.
