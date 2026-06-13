Plan:
1. Open `src/llama-kv-cells.h`
2. Add `_used_count`, `_used_min`, `_used_max_p1` private fields and initialize them.
3. Add `used_insert(uint32_t i)` and `used_erase(uint32_t i)` private methods.
4. Replace `std::set<uint32_t> used;` with these fields.
5. In `clear()`, `reset()`, reset these fields instead of `used.clear()`.
6. Update `used.insert()` and `used.erase()` calls across the file.
7. Make sure `get_used()`, `used_min()`, and `used_max_p1()` return the values of `_used_count`, `_used_min`, and `_used_max_p1`.
