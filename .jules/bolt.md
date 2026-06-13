## 2024-05-24 - Avoid std::set overhead in llama_kv_cells
**Learning:** `std::set` uses a red-black tree, leading to allocation overhead. For tracking KV cell usage in `llama_kv_cells` (`std::set<uint32_t> used`), this is inefficient during `llama_kv_cache::find_slot`. The memory explicitly states to track cell usage using scalar variables (`_used_count`, `_used_min`, `_used_max_p1`) and the `pos` array (`pos[i] != -1`) instead of `std::set`.
**Action:** Replace `std::set<uint32_t> used` in `llama_kv_cells` with scalar variables tracking count, min, and max position. When querying min/max, update the scalars sequentially if elements are removed at the boundaries.

## 2024-05-24 - Map Re-creation in loops
**Learning:** React/Svelte components that traverse trees using utility functions (like `getMessageSiblings` or `filterByLeafNodeId` in `branching.ts`) were repeatedly instantiating `Map` objects internally to map ID -> Node (`const nodeMap = new Map()`). Inside map loops this caused O(N^2) overhead per component render.
**Action:** Always accept an optional pre-computed `nodeMap` in utility tree traversal functions and share the Map down the call tree when processing arrays or traversing nodes.
