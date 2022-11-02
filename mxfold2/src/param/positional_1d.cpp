#include <string>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include "positional_1d.h"
#include "util.h"

namespace py = pybind11;

PositionalNearestNeighbor1D::
PositionalNearestNeighbor1D(const std::string& seq, pybind11::object obj) :
    score_paired_(::get_unchecked<1>(obj, "score_paired")),
    count_paired_(::get_mutable_unchecked<1>(obj, "count_paired")),

    score_hairpin_length_(::get_unchecked<1>(obj, "score_hairpin_length")),
    count_hairpin_length_(::get_mutable_unchecked<1>(obj, "count_hairpin_length")),
    score_bulge_length_(::get_unchecked<1>(obj, "score_bulge_length")),
    count_bulge_length_(::get_mutable_unchecked<1>(obj, "count_bulge_length")),
    score_internal_length_(::get_unchecked<1>(obj, "score_internal_length")),
    count_internal_length_(::get_mutable_unchecked<1>(obj, "count_internal_length")),
    score_internal_explicit_(::get_unchecked<2>(obj, "score_internal_explicit")),
    count_internal_explicit_(::get_mutable_unchecked<2>(obj, "count_internal_explicit")),
    score_internal_symmetry_(::get_unchecked<1>(obj, "score_internal_symmetry")),
    count_internal_symmetry_(::get_mutable_unchecked<1>(obj, "count_internal_symmetry")),
    score_internal_asymmetry_(::get_unchecked<1>(obj, "score_internal_asymmetry")),
    count_internal_asymmetry_(::get_mutable_unchecked<1>(obj, "count_internal_asymmetry")),
    score_helix_length_(::get_unchecked<1>(obj, "score_helix_length")),
    count_helix_length_(::get_mutable_unchecked<1>(obj, "count_helix_length")),

    MAX_HAIRPIN_LENGTH(score_hairpin_length_.shape(0)-1),
    MAX_BULGE_LENGTH(score_bulge_length_.shape(0)-1),
    MAX_INTERNAL_LENGTH(score_internal_length_.shape(0)-1),
    MAX_SINGLE_LENGTH(std::min(MAX_BULGE_LENGTH, MAX_INTERNAL_LENGTH)),
    MAX_INTERNAL_SYMMETRIC_LENGTH(score_internal_symmetry_.shape(0)-1),
    MAX_INTERNAL_ASYMMETRY(score_internal_asymmetry_.shape(0)-1),
    MAX_INTERNAL_EXPLICIT_LENGTH(score_internal_explicit_.shape(0)-1),
    MAX_HELIX_LENGTH(score_helix_length_.shape(0)-1)
{

}

auto
PositionalNearestNeighbor1D::
score_hairpin(size_t i, size_t j) const -> ScoreType
{
    const auto l = (j-1)-(i+1)+1;
    auto e = 0.;
    e += score_hairpin_length_[std::min<u_int32_t>(l, MAX_HAIRPIN_LENGTH)];
    e += score_paired_(i);
    e += score_paired_(j);
    return e;
}

void
PositionalNearestNeighbor1D::
count_hairpin(size_t i, size_t j, ScoreType v)
{
    const auto l = (j-1)-(i+1)+1;

#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
    count_hairpin_length_[std::min<u_int32_t>(l, 30)] += v;
#else    
    if (l <= MAX_HAIRPIN_LENGTH)
        count_hairpin_length_[l] += v;
#endif
    count_paired_(i) += v;
    count_paired_(j) += v;
}

auto
PositionalNearestNeighbor1D::
score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType
{
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);
    auto e = std::numeric_limits<ScoreType>::lowest();

    if (ll==0) // stack
    {
        auto e = score_paired_(i) + score_paired_(j);
        return e;
    }
    else if (ls==0) // bulge
    {
        auto e = score_bulge_length_[std::min<u_int16_t>(ll, MAX_BULGE_LENGTH)];
        e += score_paired_(i) + score_paired_(j);
        return e;
    }
    else // internal loop
    {
        auto e = score_internal_length_[std::min<u_int32_t>(ls+ll, MAX_INTERNAL_LENGTH)];
        e += score_internal_explicit_(std::min<u_int32_t>(ls, MAX_INTERNAL_EXPLICIT_LENGTH),
                                        std::min<u_int32_t>(ll, MAX_INTERNAL_EXPLICIT_LENGTH));
        if (ls==ll)
            e += score_internal_symmetry_[std::min<u_int32_t>(ll, MAX_INTERNAL_SYMMETRIC_LENGTH)];
        e += score_internal_asymmetry_[std::min<u_int32_t>(ll-ls, MAX_INTERNAL_ASYMMETRY)];
        e += score_paired_(i) + score_paired_(j);
        return e;
    }

    return e;
}

void
PositionalNearestNeighbor1D::
count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v)
{
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);

    if (ll==0) // stack
    {
        count_paired_(i) += v;
        count_paired_(j) += v;
    }
    else if (ls==0) // bulge
    {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
        count_bulge_length_[std::min<u_int32_t>(ll, 30)] += v;
#else
        if (ll <= MAX_BULGE_LENGTH)
            count_bulge_length_[ll] += v;
#endif
        count_paired_(i) += v;
        count_paired_(j) += v;
    }
    else // internal loop
    {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
        count_internal_length_[std::min<u_int32_t>(ls+ll, 30)] += v;
#else
        if (ls+ll <= MAX_INTERNAL_LENGTH)
            count_internal_length_[ls+ll] += v;
#endif
        count_internal_explicit_(std::min<u_int32_t>(ls, MAX_INTERNAL_EXPLICIT_LENGTH), 
                            std::min<u_int32_t>(ll, MAX_INTERNAL_EXPLICIT_LENGTH)) += v;
        if (ls==ll)
            count_internal_symmetry_[std::min<u_int32_t>(ll, MAX_INTERNAL_SYMMETRIC_LENGTH)] += v;
        count_internal_asymmetry_[std::min<u_int32_t>(ll-ls, MAX_INTERNAL_ASYMMETRY)] += v;
        count_paired_(i) += v;
        count_paired_(j) += v;
    }
}

auto
PositionalNearestNeighbor1D::
score_helix(size_t i, size_t j, size_t m) const -> ScoreType
{
    auto e = ScoreType(0.);
    for (auto k=1; k<m; k++)
    {
        e += score_paired_(i+(k-1));
        e += score_paired_(j-(k-1));
    }
    e += score_helix_length_[std::min<u_int32_t>(m, MAX_HELIX_LENGTH)];
    return e;
}

void
PositionalNearestNeighbor1D::
count_helix(size_t i, size_t j, size_t m, ScoreType v)
{
    for (auto k=1; k<m; k++)
    {
        count_paired_(i+(k-1)) += v;
        count_paired_(j-(k-1)) += v;
    }
    count_helix_length_[std::min<u_int32_t>(m, MAX_HELIX_LENGTH)] += v;
}

auto
PositionalNearestNeighbor1D::
score_multi_loop(size_t i, size_t j) const -> ScoreType
{
    return score_paired_(i) + score_paired_(j);
}

void
PositionalNearestNeighbor1D::
count_multi_loop(size_t i, size_t j, ScoreType v)
{
    count_paired_(i) += v;
    count_paired_(j) += v;
}

auto
PositionalNearestNeighbor1D::
score_multi_paired(size_t i, size_t j) const -> ScoreType
{
    return 0.;
}

void
PositionalNearestNeighbor1D::
count_multi_paired(size_t i, size_t j, ScoreType v)
{
}

auto
PositionalNearestNeighbor1D::
score_multi_unpaired(size_t i, size_t j) const -> ScoreType
{
    return 0.;
}

void
PositionalNearestNeighbor1D::
count_multi_unpaired(size_t i, size_t j, ScoreType v)
{
}

auto
PositionalNearestNeighbor1D::
score_external_paired(size_t i, size_t j) const -> ScoreType
{
    return 0.;
}

void
PositionalNearestNeighbor1D::
count_external_paired(size_t i, size_t j, ScoreType v)
{
}

auto
PositionalNearestNeighbor1D::
score_external_unpaired(size_t i, size_t j) const -> ScoreType
{
    return 0.;
}

void
PositionalNearestNeighbor1D::
count_external_unpaired(size_t i, size_t j, ScoreType v)
{
}