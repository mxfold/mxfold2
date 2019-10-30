#include <string>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include "bpscore.h"
#include "util.h"

namespace py = pybind11;

PositionalBasePairScore::
PositionalBasePairScore(const std::string& seq, pybind11::object obj) :
    score_paired_(::get_unchecked<2>(obj, "score_paired")),
    count_paired_(::get_mutable_unchecked<2>(obj, "count_paired")),
    score_unpaired_(::get_unchecked<1>(obj, "score_unpaired")),
    count_unpaired_(::get_mutable_unchecked<1>(obj, "count_unpaired"))
{
}

auto
PositionalBasePairScore::
score_paired(size_t i, size_t j) const -> ScoreType
{
    return score_paired_(i, j);
}

void
PositionalBasePairScore::
count_paired(size_t i, size_t j, ScoreType v)
{
    count_paired_(i, j) += v;
}

auto
PositionalBasePairScore::
score_unpaired(size_t i) const -> ScoreType
{
    return score_unpaired_(i);
}

void
PositionalBasePairScore::
count_unpaired(size_t i, ScoreType v)
{
    count_unpaired_(i) += v;
}
