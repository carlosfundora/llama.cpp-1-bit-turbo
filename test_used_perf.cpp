#include <iostream>
#include <vector>
#include <set>
#include <chrono>

int main() {
    std::set<uint32_t> used;
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < 1000000; i++) {
        used.insert(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "set insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < 1000000; i++) {
        used.erase(i);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "set erase: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    return 0;
}
