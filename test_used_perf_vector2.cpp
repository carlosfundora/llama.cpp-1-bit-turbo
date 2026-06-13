#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdint>

class FastUsedSet {
    std::vector<bool> is_used;
    uint32_t count = 0;

    // Optional: we can track the max used element
    // Tracking min is also possible.
    // Given the cache size is generally small (e.g. 8192, 32768, etc)
    // a linear scan to find min/max isn't terrible, but we can optimize.
};

int main() {
    return 0;
}
