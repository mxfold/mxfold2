#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include "mix.h"

namespace py = pybind11;

namespace {
template <class T>
T make_turner_with_encoding(const std::string& seq, pybind11::object obj,
                            std::shared_ptr<BaseEncoding> encoding) {
    if constexpr (std::is_same_v<T, TurnerNearestNeighbor>) {
        return T(seq, obj, encoding);
    } else {
        return T(seq, obj);
    }
}
} // namespace

template <class T, class P>
MixedNearestNeighborTempl<T, P>::
MixedNearestNeighborTempl(const std::string& seq, pybind11::object obj)
    :   turner_(seq, py::cast<py::dict>(obj)["turner"]),
        positional_(seq, py::cast<py::dict>(obj)["positional"]),
        score_weight_turner_(1.0f),
        score_weight_positional_(1.0f),
        count_weight_turner_(1.0f),
        count_weight_positional_(1.0f)
{
    auto dict = py::cast<py::dict>(obj);

    // Backward compatibility: weight_turner/weight_positional sets both score and count weights
    if (dict.contains("weight_turner")) {
        float w = py::cast<float>(dict["weight_turner"]);
        score_weight_turner_ = w;
        count_weight_turner_ = w;
    }
    if (dict.contains("weight_positional")) {
        float w = py::cast<float>(dict["weight_positional"]);
        score_weight_positional_ = w;
        count_weight_positional_ = w;
    }

    // New parameters override if specified
    if (dict.contains("weight_score_turner"))
        score_weight_turner_ = py::cast<float>(dict["weight_score_turner"]);
    if (dict.contains("weight_score_positional"))
        score_weight_positional_ = py::cast<float>(dict["weight_score_positional"]);
    if (dict.contains("weight_count_turner"))
        count_weight_turner_ = py::cast<float>(dict["weight_count_turner"]);
    if (dict.contains("weight_count_positional"))
        count_weight_positional_ = py::cast<float>(dict["weight_count_positional"]);
}

template <class T, class P>
MixedNearestNeighborTempl<T, P>::
MixedNearestNeighborTempl(const std::string& seq, pybind11::object obj,
                          std::shared_ptr<BaseEncoding> encoding)
    :   turner_(make_turner_with_encoding<T>(seq, py::cast<py::dict>(obj)["turner"], encoding)),
        positional_(seq, py::cast<py::dict>(obj)["positional"]),
        score_weight_turner_(1.0f),
        score_weight_positional_(1.0f),
        count_weight_turner_(1.0f),
        count_weight_positional_(1.0f)
{
    auto dict = py::cast<py::dict>(obj);

    // Backward compatibility: weight_turner/weight_positional sets both score and count weights
    if (dict.contains("weight_turner")) {
        float w = py::cast<float>(dict["weight_turner"]);
        score_weight_turner_ = w;
        count_weight_turner_ = w;
    }
    if (dict.contains("weight_positional")) {
        float w = py::cast<float>(dict["weight_positional"]);
        score_weight_positional_ = w;
        count_weight_positional_ = w;
    }

    // New parameters override if specified
    if (dict.contains("weight_score_turner"))
        score_weight_turner_ = py::cast<float>(dict["weight_score_turner"]);
    if (dict.contains("weight_score_positional"))
        score_weight_positional_ = py::cast<float>(dict["weight_score_positional"]);
    if (dict.contains("weight_count_turner"))
        count_weight_turner_ = py::cast<float>(dict["weight_count_turner"]);
    if (dict.contains("weight_count_positional"))
        count_weight_positional_ = py::cast<float>(dict["weight_count_positional"]);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_hairpin(size_t i, size_t j) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_hairpin(i, j)
         + score_weight_positional_ * positional_.score_hairpin(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_single_loop(i, j, k, l)
         + score_weight_positional_ * positional_.score_single_loop(i, j, k, l);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_helix(size_t i, size_t j, size_t m) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_helix(i, j, m)
         + score_weight_positional_ * positional_.score_helix(i, j, m);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_multi_loop(size_t i, size_t j) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_multi_loop(i, j)
         + score_weight_positional_ * positional_.score_multi_loop(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_multi_paired(size_t i, size_t j) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_multi_paired(i, j)
         + score_weight_positional_ * positional_.score_multi_paired(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_multi_unpaired(size_t i, size_t j) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_multi_unpaired(i, j)
         + score_weight_positional_ * positional_.score_multi_unpaired(i, j);
}


template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_external_zero() const -> ScoreType
{
    return score_weight_turner_ * turner_.score_external_zero()
         + score_weight_positional_ * positional_.score_external_zero();
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_external_paired(size_t i, size_t j) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_external_paired(i, j)
         + score_weight_positional_ * positional_.score_external_paired(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_external_unpaired(size_t i, size_t j) const -> ScoreType
{
    return score_weight_turner_ * turner_.score_external_unpaired(i, j)
         + score_weight_positional_ * positional_.score_external_unpaired(i, j);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_hairpin(size_t i, size_t j, ScoreType v)
{
    turner_.count_hairpin(i, j, v * count_weight_turner_);
    positional_.count_hairpin(i, j, v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v)
{
    turner_.count_single_loop(i, j, k, l, v * count_weight_turner_);
    positional_.count_single_loop(i, j, k, l, v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_helix(size_t i, size_t j, size_t m, ScoreType v)
{
    turner_.count_helix(i, j, m, v * count_weight_turner_);
    positional_.count_helix(i, j, m, v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_multi_loop(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_loop(i, j, v * count_weight_turner_);
    positional_.count_multi_loop(i, j, v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_multi_paired(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_paired(i, j, v * count_weight_turner_);
    positional_.count_multi_paired(i, j, v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_multi_unpaired(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_unpaired(i, j, v * count_weight_turner_);
    positional_.count_multi_unpaired(i, j, v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_external_zero(ScoreType v)
{
    turner_.count_external_zero(v * count_weight_turner_);
    positional_.count_external_zero(v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_external_paired(size_t i, size_t j, ScoreType v)
{
    turner_.count_external_paired(i, j, v * count_weight_turner_);
    positional_.count_external_paired(i, j, v * count_weight_positional_);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_external_unpaired(size_t i, size_t j, ScoreType v)
{
    turner_.count_external_unpaired(i, j, v * count_weight_turner_);
    positional_.count_external_unpaired(i, j, v * count_weight_positional_);
}

// instantiation
#include "turner.h"
#include "contrafold.h"
#include "positional.h"
#include "positional_bl.h"
#include "positional_1d.h"

template class MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor>;
template class MixedNearestNeighborTempl<CONTRAfoldNearestNeighbor, PositionalNearestNeighbor>;
template class MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighborBL>;
template class MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor1D>;
