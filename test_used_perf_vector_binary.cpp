#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> used;
    // We insert 50k randomly
    std::vector<uint32_t> to_insert;
    for (uint32_t i = 0; i < 50000; i++) {
        to_insert.push_back(rand() % 100000);
    }

    for (uint32_t i = 0; i < 50000; i++) {
        auto val = to_insert[i];
        auto it = std::lower_bound(used.begin(), used.end(), val);
        if (it == used.end() || *it != val) {
            used.insert(it, val);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "vector insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    return 0;
}
