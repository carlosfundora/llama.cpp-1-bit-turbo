#include <iostream>
#include <vector>
#include <cstdint>

std::vector<int> pos(10, -1);
uint32_t _used_count = 0;
uint32_t _used_min = UINT32_MAX;
uint32_t _used_max_p1 = 0;

void used_insert(uint32_t i) {
    _used_count++;
    if (i < _used_min) _used_min = i;
    if (i + 1 > _used_max_p1) _used_max_p1 = i + 1;
}

void used_erase(uint32_t i) {
    _used_count--;
    if (_used_count == 0) {
        _used_min = UINT32_MAX;
        _used_max_p1 = 0;
    } else {
        if (i == _used_min) {
            while (_used_min < pos.size() && pos[_used_min] == -1) {
                _used_min++;
            }
        }
        if (i + 1 == _used_max_p1) {
            while (_used_max_p1 > 0 && pos[_used_max_p1 - 1] == -1) {
                _used_max_p1--;
            }
        }
    }
}

int main() {
    auto insert = [](int i) { pos[i] = 1; used_insert(i); };
    auto erase = [](int i) { pos[i] = -1; used_erase(i); };

    insert(5);
    std::cout << _used_count << " " << _used_min << " " << _used_max_p1 << std::endl;
    insert(8);
    std::cout << _used_count << " " << _used_min << " " << _used_max_p1 << std::endl;
    erase(5);
    std::cout << _used_count << " " << _used_min << " " << _used_max_p1 << std::endl;
    erase(8);
    std::cout << _used_count << " " << (int)_used_min << " " << _used_max_p1 << std::endl;
    return 0;
}
