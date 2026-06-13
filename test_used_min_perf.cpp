#include <iostream>
#include <vector>
#include <set>
#include <chrono>

int main() {
    std::set<uint32_t> used_set;
    std::vector<bool> used_vec(100000, false);

    used_set.insert(99999);
    used_vec[99999] = true;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
        auto it = used_set.begin();
        if (it != used_set.end()) {
            volatile uint32_t val = *it;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "set min: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        for (uint32_t j = 0; j < 100000; j++) {
            if (used_vec[j]) {
                volatile uint32_t val = j;
                break;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "vec min scan: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 10 << " us" << std::endl;

    return 0;
}
