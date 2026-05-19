#include <iostream>
#include <vector>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // Bitset - using actual uint64_t bits
    std::vector<uint64_t> used(100000 / 64 + 1, 0);
    uint32_t count = 0;

    std::vector<uint32_t> to_insert;
    for (uint32_t i = 0; i < 50000; i++) {
        to_insert.push_back(rand() % 100000);
    }

    for (uint32_t i = 0; i < 50000; i++) {
        uint32_t val = to_insert[i];
        if (!(used[val / 64] & (1ULL << (val % 64)))) {
            used[val / 64] |= (1ULL << (val % 64));
            count++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "bit array insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    return 0;
}
