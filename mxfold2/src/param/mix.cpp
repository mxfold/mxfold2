#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include "turner.h"
#include "positional.h"
#include "mix.h"

namespace py = pybind11;
MixedNearestNeighbor::
MixedNearestNeighbor(const std::string& seq, pybind11::object obj)
    :   turner_(seq, py::cast<py::dict>(obj)["turner"]),
        positional_(seq, py::cast<py::dict>(obj)["positional"])
{

}

auto
MixedNearestNeighbor::
score_hairpin(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_hairpin(i, j) + positional_.score_hairpin(i, j);
}

auto
MixedNearestNeighbor::
score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType
{
    return turner_.score_single_loop(i, j, k, l) + positional_.score_single_loop(i, j, k, l);
}

auto
MixedNearestNeighbor::
score_helix(size_t i, size_t j, size_t m) const -> ScoreType
{
    return turner_.score_helix(i, j, m) + positional_.score_helix(i, j, m);
}

auto
MixedNearestNeighbor::
score_multi_loop(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_multi_loop(i, j) + positional_.score_multi_loop(i, j);
}

auto
MixedNearestNeighbor::
score_multi_paired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_multi_paired(i, j) + positional_.score_multi_paired(i, j);
}

auto
MixedNearestNeighbor::
score_multi_unpaired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_multi_unpaired(i, j) + positional_.score_multi_unpaired(i, j);
}


auto
MixedNearestNeighbor::
score_external_zero() const -> ScoreType
{
    return turner_.score_external_zero() + positional_.score_external_zero();
}

auto
MixedNearestNeighbor::
score_external_paired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_external_paired(i, j) + positional_.score_external_paired(i, j);
}

auto
MixedNearestNeighbor::
score_external_unpaired(size_t i, size_t j) const -> ScoreType
{
    return turner_.score_external_unpaired(i, j) + positional_.score_external_unpaired(i, j);
}

void
MixedNearestNeighbor::
count_hairpin(size_t i, size_t j, ScoreType v)
{
    turner_.count_hairpin(i, j, v);
    positional_.count_hairpin(i, j, v);
}

void
MixedNearestNeighbor::
count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v)
{
    turner_.count_single_loop(i, j, k, l, v);
    positional_.count_single_loop(i, j, k, l, v);
}

void
MixedNearestNeighbor::
count_helix(size_t i, size_t j, size_t m, ScoreType v)
{
    turner_.count_helix(i, j, m, v);
    positional_.count_helix(i, j, m, v);
}

void
MixedNearestNeighbor::
count_multi_loop(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_loop(i, j, v);
    positional_.count_multi_loop(i, j, v);
}

void
MixedNearestNeighbor::
count_multi_paired(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_paired(i, j, v);
    positional_.count_multi_paired(i, j, v);
}

void
MixedNearestNeighbor::
count_multi_unpaired(size_t i, size_t j, ScoreType v)
{
    turner_.count_multi_unpaired(i, j, v);
    positional_.count_multi_unpaired(i, j, v);
}

void
MixedNearestNeighbor::
count_external_zero(ScoreType v)
{
    turner_.count_external_zero(v);
    positional_.count_external_zero(v);
}

void
MixedNearestNeighbor::
count_external_paired(size_t i, size_t j, ScoreType v)
{
    turner_.count_external_paired(i, j, v);
    positional_.count_external_paired(i, j, v);
}

void
MixedNearestNeighbor::
count_external_unpaired(size_t i, size_t j, ScoreType v)
{
    turner_.count_external_unpaired(i, j, v);
    positional_.count_external_unpaired(i, j, v);
}
