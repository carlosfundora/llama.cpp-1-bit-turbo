#include <iostream>
#include <vector>
#include <set>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::set<uint32_t> used_set;
    for (uint32_t i = 0; i < 50000; i++) {
        used_set.insert(i);
        volatile uint32_t min = *used_set.begin();
        volatile uint32_t max = *used_set.rbegin() + 1;
    }
    for (uint32_t i = 0; i < 50000; i++) {
        used_set.erase(i);
        if (!used_set.empty()) {
            volatile uint32_t min = *used_set.begin();
            volatile uint32_t max = *used_set.rbegin() + 1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "set min/max: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<bool> used_vec(100000, false);
    uint32_t used_min = 100000;
    uint32_t used_max_p1 = 0;
    uint32_t count = 0;

    for (uint32_t i = 0; i < 50000; i++) {
        used_vec[i] = true;
        count++;
        if (i < used_min) used_min = i;
        if (i + 1 > used_max_p1) used_max_p1 = i + 1;

        volatile uint32_t min = used_min;
        volatile uint32_t max = used_max_p1;
    }
    for (uint32_t i = 0; i < 50000; i++) {
        used_vec[i] = false;
        count--;
        if (count == 0) {
            used_min = 100000;
            used_max_p1 = 0;
        } else {
            if (i == used_min) {
                while (!used_vec[used_min]) used_min++;
            }
            if (i + 1 == used_max_p1) {
                while (!used_vec[used_max_p1 - 1]) used_max_p1--;
            }
            volatile uint32_t min = used_min;
            volatile uint32_t max = used_max_p1;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "vec min/max: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    return 0;
}
