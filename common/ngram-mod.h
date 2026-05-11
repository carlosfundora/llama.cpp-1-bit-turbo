#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// LCG hash multiplier (same in idx() and rolling hash)
static constexpr uint64_t NGRAM_HASH_M = 6364136223846793005ULL;

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

    // --- Rolling hash support ---
    // Precomputed M^(n-1) for O(1) window slide instead of O(n) rehash.
    // Usage: h_init = hash_full(tokens); h_next = hash_roll(h_init, old_tok, new_tok)
    uint64_t hash_full(const entry_t * tokens) const;
    uint64_t hash_roll(uint64_t h_prev, entry_t old_tok, entry_t new_tok) const;
    entry_t  get_by_hash(uint64_t h) const;

    // Fused draft generation: rolling hash + optional prefetch.
    // Returns number of tokens drafted into `out` (up to max_draft).
    int draft_rolling(const entry_t * ctx, int max_draft, entry_t * out) const;

    bool is_power_of_two() const { return mask != 0; }

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;

    // Power-of-two fast path: mask = size - 1 (0 if not power-of-two)
    size_t mask;

    // M^(n-1) for rolling hash window slide
    uint64_t m_pow_n_minus_1;
};
