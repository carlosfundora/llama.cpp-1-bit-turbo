#include <iostream>
#include <vector>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // Bitset
    std::vector<bool> used(100000, false);
    uint32_t count = 0;

    std::vector<uint32_t> to_insert;
    for (uint32_t i = 0; i < 50000; i++) {
        to_insert.push_back(rand() % 100000);
    }

    for (uint32_t i = 0; i < 50000; i++) {
        if (!used[to_insert[i]]) {
            used[to_insert[i]] = true;
            count++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "bool array insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    return 0;
}
