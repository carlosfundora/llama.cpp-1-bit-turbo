#include <iostream>
#include <set>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::set<uint32_t> used;
    for (uint32_t i = 0; i < 100000; i++) {
        used.insert(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "set insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    return 0;
}
