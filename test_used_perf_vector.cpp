#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

int main() {
    std::vector<uint32_t> used;
    // Just measuring simple sequential inserts and erases which is a worst case for vectors if implemented naively,
    // but here we can keep a boolean array for constant time lookup/insert/erase,
    // and maintain the min/max or do a scan.
    std::vector<bool> is_used(1000000, false);
    uint32_t num_used = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < 1000000; i++) {
        if (!is_used[i]) {
            is_used[i] = true;
            num_used++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "bool array insert: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < 1000000; i++) {
        if (is_used[i]) {
            is_used[i] = false;
            num_used--;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "bool array erase: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    return 0;
}
