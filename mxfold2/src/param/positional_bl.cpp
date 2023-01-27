#include <string>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include "positional_bl.h"
#include "util.h"

namespace py = pybind11;

PositionalNearestNeighborBL::
PositionalNearestNeighborBL(const std::string& seq, pybind11::object obj) : 
    embedding_(::get_unchecked<2>(obj, "embedding")),
    bl_w_helix_stacking_(::get_unchecked<2>(obj, "bl_w_helix_stacking")),
    bl_b_helix_stacking_(::get_unchecked<1>(obj, "bl_b_helix_stacking")),
    bl_w_mismatch_external_(::get_unchecked<2>(obj, "bl_w_mismatch_external")),
    bl_b_mismatch_external_(::get_unchecked<1>(obj, "bl_b_mismatch_external")),
    bl_w_mismatch_hairpin_(::get_unchecked<2>(obj, "bl_w_mismatch_hairpin")),
    bl_b_mismatch_hairpin_(::get_unchecked<1>(obj, "bl_b_mismatch_hairpin")),
    bl_w_mismatch_internal_(::get_unchecked<2>(obj, "bl_w_mismatch_internal")),
    bl_b_mismatch_internal_(::get_unchecked<1>(obj, "bl_b_mismatch_internal")),
    bl_w_mismatch_multi_(::get_unchecked<2>(obj, "bl_w_mismatch_multi")),
    bl_b_mismatch_multi_(::get_unchecked<1>(obj, "bl_b_mismatch_multi")),
    bl_w_base_hairpin_(::get_unchecked<2>(obj, "bl_w_base_hairpin")),
    bl_b_base_hairpin_(::get_unchecked<1>(obj, "bl_b_base_hairpin")),
    bl_w_base_internal_(::get_unchecked<2>(obj, "bl_w_base_internal")),
    bl_b_base_internal_(::get_unchecked<1>(obj, "bl_b_base_internal")),
    bl_w_base_multi_(::get_unchecked<2>(obj, "bl_w_base_multi")),
    bl_b_base_multi_(::get_unchecked<1>(obj, "bl_b_base_multi")),
    bl_w_base_external_(::get_unchecked<2>(obj, "bl_w_base_external")),
    bl_b_base_external_(::get_unchecked<1>(obj, "bl_b_base_external")),
    score_hairpin_length_(::get_unchecked<1>(obj, "score_hairpin_length")),
    score_bulge_length_(::get_unchecked<1>(obj, "score_bulge_length")),
    score_internal_length_(::get_unchecked<1>(obj, "score_internal_length")),
    score_internal_explicit_(::get_unchecked<2>(obj, "score_internal_explicit")),
    score_internal_symmetry_(::get_unchecked<1>(obj, "score_internal_symmetry")),
    score_internal_asymmetry_(::get_unchecked<1>(obj, "score_internal_asymmetry")),
    score_helix_length_(::get_unchecked<1>(obj, "score_helix_length")),
    MAX_HAIRPIN_LENGTH(score_hairpin_length_.shape(0)-1),
    MAX_BULGE_LENGTH(score_bulge_length_.shape(0)-1),
    MAX_INTERNAL_LENGTH(score_internal_length_.shape(0)-1),
    MAX_SINGLE_LENGTH(std::min(MAX_BULGE_LENGTH, MAX_INTERNAL_LENGTH)),
    MAX_INTERNAL_SYMMETRIC_LENGTH(score_internal_symmetry_.shape(0)-1),
    MAX_INTERNAL_ASYMMETRY(score_internal_asymmetry_.shape(0)-1),
    MAX_INTERNAL_EXPLICIT_LENGTH(score_internal_explicit_.shape(0)-1),
    MAX_HELIX_LENGTH(score_helix_length_.shape(0)-1),
    cnt_(py::cast<py::dict>(obj)["cnt"])
#ifdef USE_PYSCORE
    ,
    py_score_hairpin(cnt_.attr("score_hairpin")),
    py_score_single_loop(cnt_.attr("score_single_loop")),
    py_score_helix(cnt_.attr("score_helix")),
    py_score_multi_loop(cnt_.attr("score_multi_loop")),
    py_score_multi_paired(cnt_.attr("score_multi_paired")),
    py_score_multi_unpaired(cnt_.attr("score_multi_unpaired")),
    py_score_external_paired(cnt_.attr("score_external_paired")),
    py_score_external_unpaired(cnt_.attr("score_external_unpaired"))
#endif
{

}

template < class T, class U, class V >
static
auto bilinear(const T& emb, int i, int j, const U& w, const V& b) 
{
    assert(emb.shape(1)==w.shape(0));
    assert(emb.shape(1)==w.shape(1));
    auto ret = 0.;
    for (auto k=0; k!=w.shape(0); k++) 
        for (auto l=0; l!=w.shape(1); l++)
            ret += emb(i, k) * emb(j, l) * w(k, l);
    ret += b;
    return ret;
}

auto
PositionalNearestNeighborBL::
score_basepair_(size_t i, size_t j) const -> ScoreType
{
    return 0.f;
}

auto
PositionalNearestNeighborBL::
score_helix_stacking_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_helix_stacking_, bl_b_helix_stacking_[0]);
}

auto
PositionalNearestNeighborBL::
score_mismatch_external_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_mismatch_external_, bl_b_mismatch_external_[0]);
}

auto
PositionalNearestNeighborBL::
score_mismatch_hairpin_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_mismatch_hairpin_, bl_b_mismatch_hairpin_[0]);
}

auto
PositionalNearestNeighborBL::
score_mismatch_internal_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_mismatch_internal_, bl_b_mismatch_internal_[0]);
}

auto
PositionalNearestNeighborBL::
score_mismatch_multi_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_mismatch_multi_, bl_b_mismatch_multi_[0]);
}

auto
PositionalNearestNeighborBL::
score_base_hairpin_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_base_hairpin_, bl_b_base_hairpin_[0]);
}

auto
PositionalNearestNeighborBL::
score_base_internal_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_base_internal_, bl_b_base_internal_[0]);
}

auto
PositionalNearestNeighborBL::
score_base_multi_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_base_multi_, bl_b_base_multi_[0]);
}

auto
PositionalNearestNeighborBL::
score_base_external_(size_t i, size_t j) const -> ScoreType
{
    return bilinear(embedding_, i, j, bl_w_base_external_, bl_b_base_external_[0]);
}

auto
PositionalNearestNeighborBL::
score_hairpin(size_t i, size_t j) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_hairpin(i, j);
    return py::cast<float>(e);
#else
    const auto l = (j-1)-(i+1)+1;
    auto e = 0.;

    e += score_hairpin_length_[std::min<u_int32_t>(l, MAX_HAIRPIN_LENGTH)];
    e += score_base_hairpin_(i+1, j-1);
    e += score_mismatch_hairpin_(i, j);
    e += score_basepair_(i, j);

    return e;
#endif
}

void
PositionalNearestNeighborBL::
count_hairpin(size_t i, size_t j, ScoreType v)
{
    cnt_.attr("count_hairpin")(i, j);
}

auto
PositionalNearestNeighborBL::
score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_single_loop(i, j, k, l);
    return py::cast<float>(e);
#else
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);
    auto e = std::numeric_limits<ScoreType>::lowest();

    if (ll==0) // stack
    {
        auto e = score_helix_stacking_(i, j);
        e += score_helix_stacking_(l, k);
        e += score_basepair_(i, j);
        return e;
    }
    else if (ls==0) // bulge
    {
        auto e = score_bulge_length_[std::min<u_int16_t>(ll, MAX_BULGE_LENGTH)];
        e += score_base_internal_(i+1, k-1) + score_base_internal_(l+1, j-1);
        e += score_mismatch_internal_(i, j) + score_mismatch_internal_(l, k);
        e += score_basepair_(i, j);
        return e;
    }
    else // internal loop
    {
        auto e = score_internal_length_[std::min<u_int32_t>(ls+ll, MAX_INTERNAL_LENGTH)];
        e += score_base_internal_(i+1, k-1) + score_base_internal_(l+1, j-1);
        e += score_internal_explicit_(std::min<u_int32_t>(ls, MAX_INTERNAL_EXPLICIT_LENGTH),
                                        std::min<u_int32_t>(ll, MAX_INTERNAL_EXPLICIT_LENGTH));
        if (ls==ll)
            e += score_internal_symmetry_[std::min<u_int32_t>(ll, MAX_INTERNAL_SYMMETRIC_LENGTH)];
        e += score_internal_asymmetry_[std::min<u_int32_t>(ll-ls, MAX_INTERNAL_ASYMMETRY)];
        e += score_mismatch_internal_(i, j) + score_mismatch_internal_(l, k);
        e += score_basepair_(i, j);
        return e;
    }

    return e;
#endif
}

void
PositionalNearestNeighborBL::
count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v)
{
    cnt_.attr("count_single_loop")(i, j, k, l);
}

auto
PositionalNearestNeighborBL::
score_helix(size_t i, size_t j, size_t m) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_helix(i, j, m);
    return py::cast<float>(e);
#else
    auto e = ScoreType(0.);
    for (auto k=1; k<m; k++)
    {
        e += score_helix_stacking_(i+(k-1), j-(k-1));
        e += score_helix_stacking_(j-k, i+k);
        e += score_basepair_(i+(k-1), j-(k-1));
    }
    e += score_helix_length_[std::min<u_int32_t>(m, MAX_HELIX_LENGTH)];
    return e;
#endif
}

void
PositionalNearestNeighborBL::
count_helix(size_t i, size_t j, size_t m, ScoreType v)
{
    cnt_.attr("count_helix")(i, j, m);
}

auto
PositionalNearestNeighborBL::
score_multi_loop(size_t i, size_t j) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_multi_loop(i, j);
    return py::cast<float>(e);
#else
    return score_mismatch_multi_(i, j) + score_basepair_(i, j);
#endif
}

void
PositionalNearestNeighborBL::
count_multi_loop(size_t i, size_t j, ScoreType v)
{
    cnt_.attr("count_multi_loop")(i, j);
}

auto
PositionalNearestNeighborBL::
score_multi_paired(size_t i, size_t j) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_multi_paired(i, j);
    return py::cast<float>(e);
#else
    return score_mismatch_multi_(j, i);
#endif
}

void
PositionalNearestNeighborBL::
count_multi_paired(size_t i, size_t j, ScoreType v)
{
    cnt_.attr("count_multi_paired")(i, j);
}

auto
PositionalNearestNeighborBL::
score_multi_unpaired(size_t i, size_t j) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_multi_unpaired(i, j);
    return py::cast<float>(e);
#else
    return score_base_multi_(i, j);
#endif
}

void
PositionalNearestNeighborBL::
count_multi_unpaired(size_t i, size_t j, ScoreType v)
{
    cnt_.attr("count_multi_unpaired")(i, j);
}

auto
PositionalNearestNeighborBL::
score_external_paired(size_t i, size_t j) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_external_paired(i, j);
    return py::cast<float>(e);
#else
    return score_mismatch_external_(j, i);
#endif
}

void
PositionalNearestNeighborBL::
count_external_paired(size_t i, size_t j, ScoreType v)
{
    cnt_.attr("count_external_paired")(i, j);
}

auto
PositionalNearestNeighborBL::
score_external_unpaired(size_t i, size_t j) const -> ScoreType
{
#ifdef USE_PYSCORE
    auto e = py_score_external_unpaired(i, j);
    return py::cast<float>(e);
#else
    return score_base_external_(i, j);
#endif
}

void
PositionalNearestNeighborBL::
count_external_unpaired(size_t i, size_t j, ScoreType v)
{
    cnt_.attr("count_external_unpaired")(i, j);
}