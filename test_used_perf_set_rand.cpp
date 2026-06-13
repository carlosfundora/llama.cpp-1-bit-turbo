#include <iostream>
#include <set>
#include <vector>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::set<uint32_t> used;
    std::vector<uint32_t> to_insert;
    for (uint32_t i = 0; i < 50000; i++) {
        to_insert.push_back(rand() % 100000);
    }

    for (uint32_t i = 0; i < 50000; i++) {
        used.insert(to_insert[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "set insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    return 0;
}
