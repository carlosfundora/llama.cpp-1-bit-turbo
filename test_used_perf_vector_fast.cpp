#include <iostream>
#include <vector>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> used;
    for (uint32_t i = 0; i < 100000; i++) {
        // finding the correct pos to insert and shifting
        auto it = std::lower_bound(used.begin(), used.end(), i);
        if (it == used.end() || *it != i) {
            used.insert(it, i);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "vector insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    return 0;
}
