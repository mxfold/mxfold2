#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include "mix.h"

namespace py = pybind11;

template <class T, class P>
MixedNearestNeighborTempl<T, P>::
MixedNearestNeighborTempl(const std::string& seq, pybind11::object obj)
    :   turner_(seq, py::cast<py::dict>(obj)["turner"]),
        positional_(seq, py::cast<py::dict>(obj)["positional"])
{

}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_hairpin(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_hairpin(i, j) + positional_.score_hairpin(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType
{
    return turner_.score_single_loop(i, j, k, l) + positional_.score_single_loop(i, j, k, l);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_helix(size_t i, size_t j, size_t m) const -> ScoreType
{
    return turner_.score_helix(i, j, m) + positional_.score_helix(i, j, m);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_multi_loop(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_multi_loop(i, j) + positional_.score_multi_loop(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_multi_paired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_multi_paired(i, j) + positional_.score_multi_paired(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_multi_unpaired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_multi_unpaired(i, j) + positional_.score_multi_unpaired(i, j);
}


template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_external_zero() const -> ScoreType
{
    return turner_.score_external_zero() + positional_.score_external_zero();
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_external_paired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_external_paired(i, j) + positional_.score_external_paired(i, j);
}

template <class T, class P>
auto
MixedNearestNeighborTempl<T, P>::
score_external_unpaired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_external_unpaired(i, j) + positional_.score_external_unpaired(i, j);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_hairpin(size_t i, size_t j, ScoreType v)
{
    turner_.count_hairpin(i, j, v);
    positional_.count_hairpin(i, j, v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v)
{
    turner_.count_single_loop(i, j, k, l, v);
    positional_.count_single_loop(i, j, k, l, v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_helix(size_t i, size_t j, size_t m, ScoreType v)
{
    turner_.count_helix(i, j, m, v);
    positional_.count_helix(i, j, m, v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_multi_loop(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_loop(i, j, v);
    positional_.count_multi_loop(i, j, v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_multi_paired(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_paired(i, j, v);
    positional_.count_multi_paired(i, j, v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_multi_unpaired(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_unpaired(i, j, v);
    positional_.count_multi_unpaired(i, j, v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_external_zero(ScoreType v)
{
    turner_.count_external_zero(v);
    positional_.count_external_zero(v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_external_paired(size_t i, size_t j, ScoreType v)
{
    turner_.count_external_paired(i, j, v);
    positional_.count_external_paired(i, j, v);
}

template <class T, class P>
void
MixedNearestNeighborTempl<T, P>::
count_external_unpaired(size_t i, size_t j, ScoreType v)
{
    turner_.count_external_unpaired(i, j, v);
    positional_.count_external_unpaired(i, j, v);
}

// instantiation
#include "turner.h"
#include "positional.h"
#include "positional_bl.h"
#include "positional_1d.h"

template class MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor>;
template class MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighborBL>;
template class MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor1D>;