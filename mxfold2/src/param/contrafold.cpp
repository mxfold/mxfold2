#include <string>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include "contrafold.h"
#include "util.h"

namespace py = pybind11;

// static
auto
CONTRAfoldNearestNeighbor::
convert_sequence(const std::string& seq) -> SeqType
{
    const auto L = seq.size();
    SeqType converted_seq(L+2);
    std::transform(std::begin(seq), std::end(seq), &converted_seq[1],
                [](auto x) {
                    switch (tolower(x)) {
                        case 'a': return 0; break;
                        case 'c': return 1; break;
                        case 'g': return 2; break;
                        case 'u':
                        case 't': return 3; break;
                        default:  return 4; break;
                    }
                });

    converted_seq[0] = converted_seq[L];
    converted_seq[L+1] = converted_seq[1];

    return converted_seq;
}

CONTRAfoldNearestNeighbor::
CONTRAfoldNearestNeighbor(const std::string& seq, pybind11::object obj) :
    seq2_(convert_sequence(seq)),
    score_base_pair_(::get_unchecked<2>(obj, "score_base_pair")),
    score_terminal_mismatch_(::get_unchecked<4>(obj, "score_terminal_mismatch")),
    score_hairpin_length_(::get_unchecked<1>(obj, "score_hairpin_length")),
    score_bulge_length_(::get_unchecked<1>(obj, "score_bulge_length")),
    score_internal_length_(::get_unchecked<1>(obj, "score_internal_length")),
    score_internal_explicit_(::get_unchecked<2>(obj, "score_internal_explicit")),
    score_internal_symmetry_(::get_unchecked<1>(obj, "score_internal_symmetry")),
    score_internal_asymmetry_(::get_unchecked<1>(obj, "score_internal_asymmetry")),
    score_bulge_0x1_(::get_unchecked<1>(obj, "score_bulge_0x1")),
    score_internal_1x1_(::get_unchecked<2>(obj, "score_internal_1x1")),
    score_helix_stacking_(::get_unchecked<4>(obj, "score_helix_stacking")),
    score_helix_closing_(::get_unchecked<2>(obj, "score_helix_closing")),
    score_multi_base_(::get_unchecked<1>(obj, "score_multi_base")),
    score_multi_unpaired_(::get_unchecked<1>(obj, "score_multi_unpaired")),
    score_multi_paired_(::get_unchecked<1>(obj, "score_multi_paired")),
    score_dangle_left_(::get_unchecked<3>(obj, "score_dangle_left")),
    score_dangle_right_(::get_unchecked<3>(obj, "score_dangle_right")),
    score_external_unpaired_(::get_unchecked<1>(obj, "score_external_unpaired")),
    score_external_paired_(::get_unchecked<1>(obj, "score_external_paired")),
    //
    count_base_pair_(::get_mutable_unchecked<2>(obj, "count_base_pair")),
    count_terminal_mismatch_(::get_mutable_unchecked<4>(obj, "count_terminal_mismatch")),
    count_hairpin_length_(::get_mutable_unchecked<1>(obj, "count_hairpin_length")),
    count_bulge_length_(::get_mutable_unchecked<1>(obj, "count_bulge_length")),
    count_internal_length_(::get_mutable_unchecked<1>(obj, "count_internal_length")),
    count_internal_explicit_(::get_mutable_unchecked<2>(obj, "count_internal_explicit")),
    count_internal_symmetry_(::get_mutable_unchecked<1>(obj, "count_internal_symmetry")),
    count_internal_asymmetry_(::get_mutable_unchecked<1>(obj, "count_internal_asymmetry")),
    count_bulge_0x1_(::get_mutable_unchecked<1>(obj, "count_bulge_0x1")),
    count_internal_1x1_(::get_mutable_unchecked<2>(obj, "count_internal_1x1")),
    count_helix_stacking_(::get_mutable_unchecked<4>(obj, "count_helix_stacking")),
    count_helix_closing_(::get_mutable_unchecked<2>(obj, "count_helix_closing")),
    count_multi_base_(::get_mutable_unchecked<1>(obj, "count_multi_base")),
    count_multi_unpaired_(::get_mutable_unchecked<1>(obj, "count_multi_unpaired")),
    count_multi_paired_(::get_mutable_unchecked<1>(obj, "count_multi_paired")),
    count_dangle_left_(::get_mutable_unchecked<3>(obj, "count_dangle_left")),
    count_dangle_right_(::get_mutable_unchecked<3>(obj, "count_dangle_right")),
    count_external_unpaired_(::get_mutable_unchecked<1>(obj, "count_external_unpaired")),
    count_external_paired_(::get_mutable_unchecked<1>(obj, "count_external_paired")),
    // 
    cache_score_hairpin_length_(score_hairpin_length_.size(), 0),
    cache_score_bulge_length_(score_bulge_length_.size(), 0),
    cache_score_internal_length_(score_internal_length_.size(), 0),
    cache_score_internal_symmetry_(score_internal_symmetry_.size(), 0),
    cache_score_internal_asymmetry_(score_internal_asymmetry_.size(), 0),
    //
    cache_score_base_pair_(score_base_pair_.shape(0), std::vector<float>(score_base_pair_.shape(1), 0)),
    cache_score_helix_stacking_(score_helix_stacking_.shape(0), std::vector<std::vector<std::vector<float>>>(score_helix_stacking_.shape(1), std::vector<std::vector<float>>(score_helix_stacking_.shape(2), std::vector<float>(score_helix_stacking_.shape(3), 0)))),
    cache_score_internal_1x1_(score_internal_1x1_.shape(0), std::vector<float>(score_internal_1x1_.shape(1), 0)),
    //
    MAX_HAIRPIN_LENGTH(score_hairpin_length_.shape(0)-1),
    MAX_BULGE_LENGTH(score_bulge_length_.shape(0)-1),
    MAX_INTERNAL_LENGTH(score_internal_length_.shape(0)-1),
    MAX_SINGLE_LENGTH(std::min(MAX_BULGE_LENGTH, MAX_INTERNAL_LENGTH)),
    MAX_INTERNAL_SYMMETRIC_LENGTH(score_internal_symmetry_.shape(0)-1),
    MAX_INTERNAL_ASYMMETRY(score_internal_asymmetry_.shape(0)-1),
    MAX_INTERNAL_EXPLICIT_LENGTH(score_internal_explicit_.shape(0)-1)
{
    cache_score_hairpin_length_[0] = score_hairpin_length_[0];
    for (auto i=1; i<score_hairpin_length_.size(); ++i)
        cache_score_hairpin_length_[i] = cache_score_hairpin_length_[i-1] + score_hairpin_length_[i];

    cache_score_bulge_length_[0] = score_bulge_length_[0];
    for (auto i=1; i<score_bulge_length_.size(); ++i)
        cache_score_bulge_length_[i] = cache_score_bulge_length_[i-1] + score_bulge_length_[i];

    cache_score_internal_length_[0] = score_internal_length_[0];
    for (auto i=1; i<score_internal_length_.size(); ++i)
        cache_score_internal_length_[i] = cache_score_internal_length_[i-1] + score_internal_length_[i];

    cache_score_internal_symmetry_[0] = score_internal_symmetry_[0];
    for (auto i=1; i<score_internal_symmetry_.size(); ++i)
        cache_score_internal_symmetry_[i] = cache_score_internal_symmetry_[i-1] + score_internal_symmetry_[i];

    cache_score_internal_asymmetry_[0] = score_internal_asymmetry_[0];
    for (auto i=1; i<score_internal_asymmetry_.size(); ++i)
        cache_score_internal_asymmetry_[i] = cache_score_internal_asymmetry_[i-1] + score_internal_asymmetry_[i];

    for (auto i=0; i<4; ++i)
        for (auto j=0; j<4; ++j)
            cache_score_base_pair_[i][j] = i<j ? score_base_pair_(i, j) : score_base_pair_(j, i);

    for (auto i=0; i<4; ++i)
        for (auto j=0; j<4; ++j)
            for (auto k=0; k<4; ++k)
                for (auto l=0; l<4; ++l)
                    cache_score_helix_stacking_[i][j][k][l] = i*1000+j*100+k*10+l < l*1000+k*100+j*10+i ? score_helix_stacking_(i, j, k, l) : score_helix_stacking_(k, l, j, i);
    
    for (auto i=0; i<cache_score_internal_1x1_.size(); ++i)
        for (auto j=0; j<cache_score_internal_1x1_[i].size(); ++j)
            cache_score_internal_1x1_[i][j] = i<j ? score_internal_1x1_(i, j) : score_internal_1x1_(j, i);
}

void
CONTRAfoldNearestNeighbor::
cache_count_base_pair(short x, short y, ScoreType v)
{
    if (x<y)
        count_base_pair_(x, y) += v;
    else
        count_base_pair_(y, x) += v;
}

void
CONTRAfoldNearestNeighbor::
cache_count_helix_stacking(short x, short y, short s, short t, ScoreType v)
{
    if (x*1000+y*100+s*10+t < t*1000+s*100+y*10+x)
        count_helix_stacking_(x, y, s, t) += v;
    else
        count_helix_stacking_(t, s, y, x) += v;
} 

void
CONTRAfoldNearestNeighbor::
cache_count_internal_1x1(short x, short y, ScoreType v)
{
    if (x<y)
        count_internal_1x1_(x, y) += v;
    else
        count_internal_1x1_(y, x) += v;
}

auto
CONTRAfoldNearestNeighbor::
score_hairpin(size_t i, size_t j) const -> ScoreType
{
    assert(i<j);
    const auto l = (j-1)-(i+1)+1;
    auto e = cache_score_hairpin_length_[std::min<u_int32_t>(l, MAX_HAIRPIN_LENGTH)];
    if (l > 3)
    {
        e += score_helix_closing_(seq2_[i], seq2_[j]);
        e += score_terminal_mismatch_(seq2_[i], seq2_[j], seq2_[i+1], seq2_[j-1]);
    }
    return e;
}

void
CONTRAfoldNearestNeighbor::
count_hairpin(size_t i, size_t j, ScoreType v)
{
    assert(i<j);
    const auto l = (j-1)-(i+1)+1;

#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
    for (auto k=0; k<=std::min<u_int32_t>(l, MAX_HAIRPIN_LENGTH); ++k)
        count_hairpin_length_[k] += v;
#else
    if (l <= MAX_HAIRPIN_LENGTH)
        for (auto k=0; k<=l; ++k)
            count_hairpin_length_[k] += v;
#endif

    if (l > 3) 
    {
        count_helix_closing_(seq2_[i], seq2_[j]) += v;
        count_terminal_mismatch_(seq2_[i], seq2_[j], seq2_[i+1], seq2_[j-1]) += v;
    }
}

auto
CONTRAfoldNearestNeighbor::
score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType
{
    assert(i<j);
    assert(k<j);
    assert(k<l);
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);
    auto e = std::numeric_limits<ScoreType>::lowest();

    if (ll==0) // stack
        return score_helix(i, j, 2);
    else if (ls==0) // bulge
    {
        auto e = cache_score_bulge_length_[std::min<u_int32_t>(ll, MAX_BULGE_LENGTH)];
        if (l1==1 && l2==0)
            e += score_bulge_0x1_(seq2_[i+1]);
        else if (l1==0 && l2==1)
            e += score_bulge_0x1_(seq2_[j-1]);

        e += score_terminal_mismatch_(seq2_[i], seq2_[j], seq2_[i+1], seq2_[j-1]);
        e += score_helix_closing_(seq2_[i], seq2_[j]);
        e += score_terminal_mismatch_(seq2_[l], seq2_[k], seq2_[l+1], seq2_[k-1]);
        e += score_helix_closing_(seq2_[l], seq2_[k]);
        e += cache_score_base_pair_[seq2_[l]][seq2_[k]];
        return e;
    }
    else // internal loop
    {
        auto e = cache_score_internal_length_[std::min<u_int32_t>(ls+ll, MAX_INTERNAL_LENGTH)];
        if (ls<=MAX_INTERNAL_EXPLICIT_LENGTH && ll<=MAX_INTERNAL_EXPLICIT_LENGTH)
            e += score_internal_explicit_(ls, ll);
        if (ls==ll)
            e += cache_score_internal_symmetry_[std::min<u_int32_t>(ll, MAX_INTERNAL_SYMMETRIC_LENGTH)];
        e += cache_score_internal_asymmetry_[std::min<u_int32_t>(ll-ls, MAX_INTERNAL_ASYMMETRY)];
        if (l1==1 && l2==1)
            e += cache_score_internal_1x1_[seq2_[i+1]][seq2_[j-1]];

        e += score_terminal_mismatch_(seq2_[i], seq2_[j], seq2_[i+1], seq2_[j-1]);
        e += score_helix_closing_(seq2_[i], seq2_[j]);
        e += score_terminal_mismatch_(seq2_[l], seq2_[k], seq2_[l+1], seq2_[k-1]);
        e += score_helix_closing_(seq2_[l], seq2_[k]);
        e += cache_score_base_pair_[seq2_[l]][seq2_[k]];
        return e;
    }
    return e;
}

void
CONTRAfoldNearestNeighbor::
count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v)
{
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);

    if (ll==0) // stack
        count_helix(i, j, 2, v);
    else if (ls==0) // bulge
    {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
        for (auto k=0; k<=std::min<u_int32_t>(ll, MAX_BULGE_LENGTH); ++k)
            count_bulge_length_[k] += v;
#else
        if (ll <= MAX_BULGE_LENGTH)
            for (auto k=0; k<=ll; ++k)
                count_bulge_length_[k] += v;
#endif
        if (l1==1 && l2==0)
            count_bulge_0x1_(seq2_[i+1]) += v;
        else if (l1==0 && l2==1)
            count_bulge_0x1_(seq2_[j-1]) += v;

        count_terminal_mismatch_(seq2_[i], seq2_[j], seq2_[i+1], seq2_[j-1]) += v;
        count_helix_closing_(seq2_[i], seq2_[j]) += v;
        count_terminal_mismatch_(seq2_[l], seq2_[k], seq2_[l+1], seq2_[k-1]) += v;
        count_helix_closing_(seq2_[l], seq2_[k]) += v;
        cache_count_base_pair(seq2_[l], seq2_[k], v);
    }
    else // internal loop
    {
        for (auto k=0; k<=std::min<u_int32_t>(ls+ll, MAX_INTERNAL_LENGTH); ++k)
            count_internal_length_[k] += v;
        if (ls<=MAX_INTERNAL_EXPLICIT_LENGTH && ll<=MAX_INTERNAL_EXPLICIT_LENGTH)
            count_internal_explicit_(ls, ll) += v;
        if (ls==ll)
        {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
            for (auto k=0; k<=std::min<u_int32_t>(ll, MAX_INTERNAL_SYMMETRIC_LENGTH); ++k)
                count_internal_symmetry_[k] += v;
#else
            if (ll <= MAX_INTERNAL_SYMMETRIC_LENGTH)
                for (auto k=0; k<=ll; ++k)
                    count_internal_symmetry_[k] += v;
#endif
        }
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
        for (auto k=0; k<=std::min<u_int32_t>(ll-ls, MAX_INTERNAL_ASYMMETRY); ++k)
            count_internal_asymmetry_[k] += v;
#else
        if (ll-ls <= MAX_INTERNAL_ASYMMETRY)
            for (auto k=0; k<=ll-ls; ++k)
                count_internal_asymmetry_[k] += v;
#endif
        if (l1==1 && l2==1)
            cache_count_internal_1x1(seq2_[i+1], seq2_[j-1], v);

        count_terminal_mismatch_(seq2_[i], seq2_[j], seq2_[i+1], seq2_[j-1]) += v;
        count_helix_closing_(seq2_[i], seq2_[j]) += v;
        count_terminal_mismatch_(seq2_[l], seq2_[k], seq2_[l+1], seq2_[k-1]) += v;
        count_helix_closing_(seq2_[l], seq2_[k]) += v;
        cache_count_base_pair(seq2_[l], seq2_[k], v);
    }
}

auto
CONTRAfoldNearestNeighbor::
score_helix(size_t i, size_t j, size_t m) const -> ScoreType
{
    auto e = ScoreType(0.);
    for (auto k=1; k<m; k++)
    {
        e += cache_score_helix_stacking_[seq2_[i+(k-1)]][seq2_[j-(k-1)]][seq2_[i+k]][seq2_[j-k]];
        e += cache_score_base_pair_[seq2_[i+k]][seq2_[j-k]];
    }
    return e;
}

void
CONTRAfoldNearestNeighbor::
count_helix(size_t i, size_t j, size_t m, ScoreType v)
{
    for (auto k=1; k<m; k++)
    {
        cache_count_helix_stacking(seq2_[i+(k-1)], seq2_[j-(k-1)], seq2_[i+k], seq2_[j-k], v);
        cache_count_base_pair(seq2_[i+k], seq2_[j-k], v);
    }
}

auto
CONTRAfoldNearestNeighbor::
score_multi_loop(size_t i, size_t j) const -> ScoreType
{
    const auto L = seq2_.size()-2;
    auto e = score_helix_closing_(seq2_[i], seq2_[j]);
    if (i+1<=L)
        e += score_dangle_left_(seq2_[i], seq2_[j], seq2_[i+1]);
    if (j-1>=1)
        e += score_dangle_right_(seq2_[i], seq2_[j], seq2_[j-1]);
    e += score_multi_paired_[0];
    e += score_multi_base_[0];

    return e;
}

void
CONTRAfoldNearestNeighbor::
count_multi_loop(size_t i, size_t j, ScoreType v)
{
    const auto L = seq2_.size()-2;
    count_helix_closing_(seq2_[i], seq2_[j]) += v;
    if (i+1<=L)
        count_dangle_left_(seq2_[i], seq2_[j], seq2_[i+1]) += v;
    if (j-1>=1)
        count_dangle_right_(seq2_[i], seq2_[j], seq2_[j-1]) += v;
    count_multi_paired_[0] += v;
    count_multi_base_[0] += v;
}

auto
CONTRAfoldNearestNeighbor::
score_multi_paired(size_t i, size_t j) const -> ScoreType
{
    const auto L = seq2_.size()-2;
    auto e = score_helix_closing_(seq2_[j], seq2_[i]);
    if (j+1<=L)
        e += score_dangle_left_(seq2_[j], seq2_[i], seq2_[j+1]);
    if (i-1>=1)
        e += score_dangle_right_(seq2_[j], seq2_[i], seq2_[i-1]);
    e += cache_score_base_pair_[seq2_[j]][seq2_[i]];
    e += score_multi_paired_[0];

    return e;
}

void
CONTRAfoldNearestNeighbor::
count_multi_paired(size_t i, size_t j, ScoreType v)
{
    const auto L = seq2_.size()-2;
    count_helix_closing_(seq2_[j], seq2_[i]) += v;
    if (j+1<=L)
        count_dangle_left_(seq2_[j], seq2_[i], seq2_[j+1]) += v;
    if (i-1>=1)
        count_dangle_right_(seq2_[j], seq2_[i], seq2_[i-1]) += v;
    cache_count_base_pair(seq2_[j], seq2_[i], v);
    count_multi_paired_[0] += v;
}

auto
CONTRAfoldNearestNeighbor::
score_multi_unpaired(size_t i, size_t j) const -> ScoreType
{
    return score_multi_unpaired_[0] * (j-i+1);
}

void
CONTRAfoldNearestNeighbor::
count_multi_unpaired(size_t i, size_t j, ScoreType v)
{
    count_multi_unpaired_[0] += (j-i+1) * v;
}

auto
CONTRAfoldNearestNeighbor::
score_external_paired(size_t i, size_t j) const -> ScoreType
{
    const auto L = seq2_.size()-2;
    auto e = score_helix_closing_(seq2_[j], seq2_[i]);
    if (j+1<=L)
        e += score_dangle_left_(seq2_[j], seq2_[i], seq2_[j+1]);
    if (i-1>=1)
        e += score_dangle_right_(seq2_[j], seq2_[i], seq2_[i-1]);
    e += cache_score_base_pair_[seq2_[j]][seq2_[i]];
    e += score_external_paired_[0];

    return e;
}

void
CONTRAfoldNearestNeighbor::
count_external_paired(size_t i, size_t j, ScoreType v)
{
    const auto L = seq2_.size()-2;
    count_helix_closing_(seq2_[j], seq2_[i]) += v;
    if (j+1<=L)
        count_dangle_left_(seq2_[j], seq2_[i], seq2_[j+1]) += v;
    if (i-1>=1)
        count_dangle_right_(seq2_[j], seq2_[i], seq2_[i-1]) += v;
    cache_count_base_pair(seq2_[j], seq2_[i], v);
    count_external_paired_[0] += v;
}

auto
CONTRAfoldNearestNeighbor::
score_external_unpaired(size_t i, size_t j) const -> ScoreType
{
    return score_external_unpaired_[0];
}

void
CONTRAfoldNearestNeighbor::
count_external_unpaired(size_t i, size_t j, ScoreType v)
{
    count_external_unpaired_[0] += v;
}