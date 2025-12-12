#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <cstdint>

namespace py = pybind11;

/**
 * Compare base pair sequences and compute confusion matrix metrics.
 *
 * @return tuple of (tp, tn, fp, fn)
 */

// Mode A: ref is a list of base pairs [(i1, j1), (i2, j2), ...]
// Accepts Python list directly to avoid conversion overhead
std::tuple<int64_t, int64_t, int64_t, int64_t>
compare_bpseq_pairs(
    py::list ref_pairs,
    py::list pred,
    int64_t L
);

// Mode B: ref is a 1D array where ref[i] = j means position i pairs with j
// Accepts Python list directly to avoid conversion overhead
std::tuple<int64_t, int64_t, int64_t, int64_t>
compare_bpseq_array(
    py::list ref,
    py::list pred
);
