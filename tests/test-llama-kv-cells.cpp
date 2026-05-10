#include "llama-kv-cells.h"
#include "testing.h"

#include <vector>
#include <cstdint>

int main() {
    testing t;

    t.test("resize and size", [&](testing & t) {
        llama_kv_cells cells;
        t.assert_equal(0u, cells.size());

        cells.resize(10);
        t.assert_equal(10u, cells.size());
        t.assert_equal(0u, cells.get_used());

        for (uint32_t i = 0; i < 10; ++i) {
            t.assert_true(cells.is_empty(i));
        }
    });

    t.test("pos_set and is_empty", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);

        cells.pos_set(5, 100);
        t.assert_true(!cells.is_empty(5));
        t.assert_equal(100, cells.pos_get(5));
        t.assert_equal(1u, cells.get_used());
        t.assert_equal(5u, cells.used_min());
        t.assert_equal(6u, cells.used_max_p1());
    });

    t.test("seq_add and seq_has", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 100);

        cells.seq_add(5, 1);
        t.assert_true(cells.seq_has(5, 1));
        t.assert_true(!cells.seq_has(5, 2));
        t.assert_equal(1, cells.seq_count(5));
        t.assert_equal(1, cells.seq_get(5));

        cells.seq_add(5, 2);
        t.assert_true(cells.seq_has(5, 2));
        t.assert_equal(2, cells.seq_count(5));
    });

    t.test("seq_pos_min/max", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);

        cells.pos_set(1, 10);
        cells.seq_add(1, 1);

        cells.pos_set(2, 20);
        cells.seq_add(2, 1);

        cells.pos_set(3, 15);
        cells.seq_add(3, 1);

        t.assert_equal(10, cells.seq_pos_min(1));
        t.assert_equal(20, cells.seq_pos_max(1));

        t.assert_equal(-1, cells.seq_pos_min(2));
    });

    t.test("rm", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 100);
        cells.seq_add(5, 1);

        cells.rm(5);
        t.assert_true(cells.is_empty(5));
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(-1, cells.seq_pos_min(1));
    });

    t.test("seq_rm", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 100);
        cells.seq_add(5, 1);
        cells.seq_add(5, 2);

        bool empty = cells.seq_rm(5, 1);
        t.assert_true(!empty);
        t.assert_true(!cells.is_empty(5));
        t.assert_true(!cells.seq_has(5, 1));
        t.assert_true(cells.seq_has(5, 2));

        empty = cells.seq_rm(5, 2);
        t.assert_true(empty);
        t.assert_true(cells.is_empty(5));
    });

    t.test("seq_keep", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 100);
        cells.seq_add(5, 1);
        cells.seq_add(5, 2);

        bool empty = cells.seq_keep(5, 1);
        t.assert_true(!empty);
        t.assert_true(cells.seq_has(5, 1));
        t.assert_true(!cells.seq_has(5, 2));

        empty = cells.seq_keep(5, 3);
        t.assert_true(empty);
        t.assert_true(cells.is_empty(5));
    });

    t.test("pos_add and pos_div", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 100);
        cells.seq_add(5, 1);

        t.assert_true(!cells.get_has_shift());

        cells.pos_add(5, 10);
        t.assert_equal(110, cells.pos_get(5));
        t.assert_equal(10, cells.get_shift(5));
        t.assert_true(cells.get_has_shift());
        t.assert_equal(110, cells.seq_pos_min(1));

        cells.pos_div(5, 2);
        t.assert_equal(55, cells.pos_get(5));
        t.assert_equal(10 + (110 - 55), cells.get_shift(5));
        t.assert_equal(55, cells.seq_pos_min(1));

        cells.reset_shift();
        t.assert_true(!cells.get_has_shift());
        t.assert_equal(0, cells.get_shift(5));
    });

    t.test("pos_add negative overflow", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 10);
        cells.seq_add(5, 1);

        bool empty = cells.pos_add(5, -20);
        t.assert_true(empty);
        t.assert_true(cells.is_empty(5));
        t.assert_equal(0u, cells.get_used());
    });

    t.test("cp and set", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 100);
        cells.seq_add(5, 1);
        cells.ext_set(5, {1, 2});

        llama_kv_cells subset = cells.cp(5, 1);
        t.assert_equal(1u, subset.size());
        t.assert_equal(100, subset.pos_get(0));
        t.assert_true(subset.seq_has(0, 1));
        t.assert_equal(1, subset.ext_get(0).x);
        t.assert_equal(2, subset.ext_get(0).y);

        llama_kv_cells other;
        other.resize(10);
        other.set(0, subset);
        t.assert_equal(100, other.pos_get(0));
        t.assert_true(other.seq_has(0, 1));
        t.assert_equal(1, other.ext_get(0).x);
        t.assert_equal(2, other.ext_get(0).y);
        t.assert_equal(100, other.seq_pos_min(1));
    });

    t.test("pos_in", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);
        cells.pos_set(5, 100);

        t.assert_true(cells.pos_in(5, 50, 150));
        t.assert_true(!cells.pos_in(5, 150, 200));
        t.assert_true(!cells.pos_in(5, 0, 100));
    });

    t.test("llama_kv_cell_ext is_2d_gt", [&](testing & t) {
        llama_kv_cell_ext e1 = {1, 1};
        llama_kv_cell_ext e2 = {2, 1};
        llama_kv_cell_ext e3 = {1, 2};

        t.assert_true(e2.is_2d_gt(1, 1));
        t.assert_true(!e1.is_2d_gt(2, 1));
        t.assert_true(e3.is_2d_gt(1, 1));
        t.assert_true(e3.is_2d_gt(2, 1));
    });

    return t.summary();
}
