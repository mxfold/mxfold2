#include "compbpseq.h"
#include <algorithm>
#include <unordered_set>

namespace {
    // Custom hash function for pair
    struct PairHash {
        std::size_t operator()(const std::pair<int64_t, int64_t>& p) const {
            // Combine hashes using a simple technique
            auto h1 = std::hash<int64_t>{}(p.first);
            auto h2 = std::hash<int64_t>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };
}

std::tuple<int64_t, int64_t, int64_t, int64_t>
compare_bpseq_pairs(
    py::list ref_pairs,
    py::list pred,
    int64_t L
) {
    // Normalize ref_pairs to set with (min, max) ordering
    // Convert directly from Python list
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> ref_set;
    for (auto item : ref_pairs) {
        py::sequence seq = item.cast<py::sequence>();
        int64_t i = seq[0].cast<int64_t>();
        int64_t j = seq[1].cast<int64_t>();
        ref_set.insert({std::min(i, j), std::max(i, j)});
    }

    // Build pred set from array: (i, pred[i]) where i < pred[i]
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> pred_set;
    int64_t pred_size = static_cast<int64_t>(py::len(pred));
    for (int64_t i = 0; i < pred_size; ++i) {
        int64_t j = pred[i].cast<int64_t>();
        if (i < j) {
            pred_set.insert({i, j});
        }
    }

    // TP: intersection of ref and pred
    int64_t tp = 0;
    for (const auto& p : pred_set) {
        if (ref_set.count(p)) {
            ++tp;
        }
    }

    // FP: pred - ref
    int64_t fp = static_cast<int64_t>(pred_set.size()) - tp;

    // FN: ref - pred
    int64_t fn = static_cast<int64_t>(ref_set.size()) - tp;

    // TN: total possible pairs - (TP + FP + FN)
    int64_t tn = L * (L - 1) / 2 - tp - fp - fn;

    return {tp, tn, fp, fn};
}

std::tuple<int64_t, int64_t, int64_t, int64_t>
compare_bpseq_array(
    py::list ref,
    py::list pred
) {
    int64_t ref_size = static_cast<int64_t>(py::len(ref));
    int64_t L = ref_size - 1;
    int64_t tp = 0, fp = 0, fn = 0;

    for (int64_t i = 0; i < ref_size; ++i) {
        int64_t j1 = ref[i].cast<int64_t>();
        int64_t j2 = pred[i].cast<int64_t>();

        if (j1 > 0 && i < j1) {  // ref has a positive pair at position i
            if (j1 == j2) {
                ++tp;
            } else if (j2 > 0 && i < j2) {
                ++fp;
                ++fn;
            } else {
                ++fn;
            }
        } else if (j2 > 0 && i < j2) {
            ++fp;
        }
    }

    int64_t tn = L * (L - 1) / 2 - tp - fp - fn;
    return {tp, tn, fp, fn};
}
