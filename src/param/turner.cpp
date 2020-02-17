#include <string>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include "turner.h"
#include "util.h"

namespace py = pybind11;

// static
auto
TurnerNearestNeighbor::
convert_sequence(const std::string& seq) -> SeqType
{
    const auto L = seq.size();
    SeqType converted_seq(L+2);
    std::transform(std::begin(seq), std::end(seq), &converted_seq[1],
                [](auto x) {
                    switch (tolower(x)) {
                        default:  return 0; break;
                        case 'a': return 1; break;
                        case 'c': return 2; break;
                        case 'g': return 3; break;
                        case 'u':
                        case 't': return 4; break;
                    }
                });

    converted_seq[0] = converted_seq[L];
    converted_seq[L+1] = converted_seq[1];

    return converted_seq;
}

// static 
int
TurnerNearestNeighbor::
complement_pair[5][5] = {
   // _  A  C  G  U 
    { 0, 0, 0, 0, 0 }, // _
    { 0, 0, 0, 0, 5 }, // A
    { 0, 0, 0, 1, 0 }, // C
    { 0, 0, 2, 0, 3 }, // G
    { 0, 6, 0, 4, 0 }, // U
};

TurnerNearestNeighbor::
TurnerNearestNeighbor(const std::string& seq, pybind11::object obj) :
    seq2_(convert_sequence(seq)),
#if 0    
    use_score_hairpin_at_least_(py::hasattr(obj, "score_hairpin_at_least")),
    use_score_bulge_at_least_(py::hasattr(obj, "score_bulge_at_least")),
    use_score_internal_at_least_(py::hasattr(obj, "score_internal_at_least")),
#else
    use_score_hairpin_at_least_(obj.contains("score_hairpin_at_least")),
    use_score_bulge_at_least_(obj.contains("score_bulge_at_least")),
    use_score_internal_at_least_(obj.contains("score_internal_at_least")),
#endif    
    score_stack_(::get_unchecked<2>(obj, "score_stack")),
    score_hairpin_(::get_unchecked<1>(obj, use_score_hairpin_at_least_ ? "score_hairpin_at_least" : "score_hairpin")),
    score_bulge_(::get_unchecked<1>(obj, use_score_bulge_at_least_ ? "score_bulge_at_least" : "score_bulge")),
    score_internal_(::get_unchecked<1>(obj, use_score_internal_at_least_ ? "score_internal_at_least" : "score_internal")),
    score_mismatch_external_(::get_unchecked<3>(obj, "score_mismatch_external")),
    score_mismatch_hairpin_(::get_unchecked<3>(obj, "score_mismatch_hairpin")),
    score_mismatch_internal_(::get_unchecked<3>(obj, "score_mismatch_internal")),
    score_mismatch_internal_1n_(::get_unchecked<3>(obj, "score_mismatch_internal_1n")),
    score_mismatch_internal_23_(::get_unchecked<3>(obj, "score_mismatch_internal_23")),
    score_mismatch_multi_(::get_unchecked<3>(obj, "score_mismatch_multi")),
    score_int11_(::get_unchecked<4>(obj, "score_int11")),
    score_int21_(::get_unchecked<5>(obj, "score_int21")),
    score_int22_(::get_unchecked<6>(obj, "score_int22")),
    score_dangle5_(::get_unchecked<2>(obj, "score_dangle5")),
    score_dangle3_(::get_unchecked<2>(obj, "score_dangle3")),
    score_ml_base_(::get_unchecked<1>(obj, "score_ml_base")),
    score_ml_closing_(::get_unchecked<1>(obj, "score_ml_closing")),
    score_ml_intern_(::get_unchecked<1>(obj, "score_ml_intern")),
    score_ninio_(::get_unchecked<1>(obj, "score_ninio")),
    score_max_ninio_(::get_unchecked<1>(obj, "score_max_ninio")),
    score_duplex_init_(::get_unchecked<1>(obj, "score_duplex_init")),
    score_terminalAU_(::get_unchecked<1>(obj, "score_terminalAU")),
    score_lxc_(::get_unchecked<1>(obj, "score_lxc")),
#if 0
    use_count_hairpin_at_least_(py::hasattr(obj, "count_hairpin_at_least")),
    use_count_bulge_at_least_(py::hasattr(obj, "count_bulge_at_least")),
    use_count_internal_at_least_(py::hasattr(obj, "count_internal_at_least")),
#else
    use_count_hairpin_at_least_(obj.contains("count_hairpin_at_least")),
    use_count_bulge_at_least_(obj.contains("count_bulge_at_least")),
    use_count_internal_at_least_(obj.contains("count_internal_at_least")),
#endif
    count_stack_(::get_mutable_unchecked<2>(obj, "count_stack")),
    count_hairpin_(::get_mutable_unchecked<1>(obj, use_count_hairpin_at_least_ ? "count_hairpin_at_least" : "count_hairpin")),
    count_bulge_(::get_mutable_unchecked<1>(obj, use_count_bulge_at_least_ ? "count_bulge_at_least" : "count_bulge")),
    count_internal_(::get_mutable_unchecked<1>(obj, use_count_internal_at_least_ ? "count_internal_at_least" : "count_internal")),
    count_mismatch_external_(::get_mutable_unchecked<3>(obj, "count_mismatch_external")),
    count_mismatch_hairpin_(::get_mutable_unchecked<3>(obj, "count_mismatch_hairpin")),
    count_mismatch_internal_(::get_mutable_unchecked<3>(obj, "count_mismatch_internal")),
    count_mismatch_internal_1n_(::get_mutable_unchecked<3>(obj, "count_mismatch_internal_1n")),
    count_mismatch_internal_23_(::get_mutable_unchecked<3>(obj, "count_mismatch_internal_23")),
    count_mismatch_multi_(::get_mutable_unchecked<3>(obj, "count_mismatch_multi")),
    count_int11_(::get_mutable_unchecked<4>(obj, "count_int11")),
    count_int21_(::get_mutable_unchecked<5>(obj, "count_int21")),
    count_int22_(::get_mutable_unchecked<6>(obj, "count_int22")),
    count_dangle5_(::get_mutable_unchecked<2>(obj, "count_dangle5")),
    count_dangle3_(::get_mutable_unchecked<2>(obj, "count_dangle3")),
    count_ml_base_(::get_mutable_unchecked<1>(obj, "count_ml_base")),
    count_ml_closing_(::get_mutable_unchecked<1>(obj, "count_ml_closing")),
    count_ml_intern_(::get_mutable_unchecked<1>(obj, "count_ml_intern")),
    count_ninio_(::get_mutable_unchecked<1>(obj, "count_ninio")),
    count_max_ninio_(::get_mutable_unchecked<1>(obj, "count_max_ninio")),
    count_duplex_init_(::get_mutable_unchecked<1>(obj, "count_duplex_init")),
    count_terminalAU_(::get_mutable_unchecked<1>(obj, "count_terminalAU")),
    count_lxc_(::get_mutable_unchecked<1>(obj, "count_lxc")),
    cache_score_hairpin_(score_hairpin_.size(), 0),
    cache_score_bulge_(score_bulge_.size(), 0),
    cache_score_internal_(score_internal_.size(), 0)
{
    if (use_score_hairpin_at_least_)
    {
        for (auto i=0; i<4; ++i)
            cache_score_hairpin_[i] = score_hairpin_[i];
        for (auto i=4; i<score_hairpin_.size(); ++i)
            cache_score_hairpin_[i] = cache_score_hairpin_[i-1] + score_hairpin_[i];
    } 
    else
    {
        for (auto i=0; i<score_hairpin_.size(); ++i)
            cache_score_hairpin_[i] = score_hairpin_[i];
    }

    if (use_score_bulge_at_least_)
    {
        for (auto i=0; i<2; ++i)
            cache_score_bulge_[i] = score_bulge_[i];
        for (auto i=2; i<score_bulge_.size(); ++i)
            cache_score_bulge_[i] = cache_score_bulge_[i-1] + score_bulge_[i];
    } 
    else
    {
        for (auto i=0; i<score_bulge_.size(); ++i)
            cache_score_bulge_[i] = score_bulge_[i];
    }

    if (use_score_internal_at_least_)
    {
        for (auto i=0; i<3; ++i)
            cache_score_internal_[i] = score_internal_[i];
        for (auto i=3; i<score_internal_.size(); ++i)
            cache_score_internal_[i] = cache_score_internal_[i-1] + score_internal_[i];
    } 
    else
    {
        for (auto i=0; i<score_internal_.size(); ++i)
            cache_score_internal_[i] = score_internal_[i];
    }
}

auto
TurnerNearestNeighbor::
score_hairpin(size_t i, size_t j) const -> ScoreType
{
    const auto l = (j-1)-(i+1)+1;
    auto e = 0.;

    if (l <= 30)
        e += cache_score_hairpin_[l];
    else
        e += cache_score_hairpin_[30] + (score_lxc_[0] * log(l / 30.));

    if (l < 3) return e;

    const auto type = complement_pair[seq2_[i]][seq2_[j]];
    if (l == 3)
        e += type > 2 ? score_terminalAU_[0] : 0;
    else
        e += score_mismatch_hairpin_(type, seq2_[i+1], seq2_[j-1]);

    return e;
}

void
TurnerNearestNeighbor::
count_hairpin(size_t i, size_t j, ScoreType v)
{
    const auto l = (j-1)-(i+1)+1;

    if (use_count_hairpin_at_least_)
    {
        if (l <= 30)
            for (auto k=l; k>=3; --k) count_hairpin_[k] += v;
        else
        {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
            for (auto k=30; k>=3; --k) count_hairpin_[k] += v;
            count_lxc_[0] += v * log(l / 30.);
#endif
        }
    }
    else
    {
        if (l <= 30)
            count_hairpin_[l] += v;
        else
        {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
            count_hairpin_[30] += v;
            count_lxc_[0] += v * log(l / 30.);
#endif
        }
    }

    if (l < 3) return;

    const auto type = complement_pair[seq2_[i]][seq2_[j]];
    if (l == 3)
    {
        if (type > 2)        
            count_terminalAU_[0] += v;
    }
    else
        count_mismatch_hairpin_(type, seq2_[i+1], seq2_[j-1]) += v;
}

auto
TurnerNearestNeighbor::
score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType
{
    const auto type1 = complement_pair[seq2_[i]][seq2_[j]];
    const auto type2 = complement_pair[seq2_[l]][seq2_[k]];
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);
    auto e = std::numeric_limits<ScoreType>::lowest();

    if (ll==0) // stack
        return score_stack_(type1, type2);
    else if (ls==0) // bulge
    {
        auto e = ll<=30 ? cache_score_bulge_[ll] : cache_score_bulge_[30] + (score_lxc_[0] * log(ll / 30.));
        if (ll==1) 
            e += score_stack_(type1, type2);
        else
        {
            if (type1 > 2)
                e += score_terminalAU_[0];
            if (type2 > 2)
                e += score_terminalAU_[0];
        }
        return e;
    }
    else // internal loop
    {
        if (ll==1 && ls==1) // 1x1 loop
            return score_int11_(type1, type2, seq2_[i+1], seq2_[j-1]);
        else if (l1==2 && l2==1) // 2x1 loop
            return score_int21_(type2, type1, seq2_[l+1], seq2_[i+1], seq2_[k-1]);
        else if (l1==1 && l2==2) // 1x2 loop
            return score_int21_(type1, type2, seq2_[i+1], seq2_[l+1], seq2_[j-1]);
        else if (ls==1) // 1xn loop
        {
            auto e = ll+1 <= 30 ? cache_score_internal_[ll+1] : cache_score_internal_[30] + (score_lxc_[0] * log((ll+1) / 30.));
            e += std::max(score_max_ninio_[0], (ll-ls) * score_ninio_[0]);
            e += score_mismatch_internal_1n_(type1, seq2_[i+1], seq2_[j-1]) + score_mismatch_internal_1n_(type2, seq2_[l+1], seq2_[k-1]);
            return e;
        }
        else if (ls==2 && ll==2) // 2x2 loop
            return score_int22_(type1, type2, seq2_[i+1], seq2_[k-1], seq2_[l+1], seq2_[j-1]);
        else if (ls==2 && ll==3) // 2x3 loop
        {
            auto e = cache_score_internal_[ls+ll] + score_ninio_[0];
            e += score_mismatch_internal_23_(type1, seq2_[i+1], seq2_[j-1]) + score_mismatch_internal_23_(type2, seq2_[l+1], seq2_[k-1]);
            return e;
        }
        else // generic internal loop
        {
            auto e = ls+ll <= 30 ? cache_score_internal_[ls+ll] : cache_score_internal_[30] + (score_lxc_[0] * log((ls+ll) / 30.));
            e += std::max(score_max_ninio_[0], (ll-ls) * score_ninio_[0]);
            e += score_mismatch_internal_(type1, seq2_[i+1], seq2_[j-1]) + score_mismatch_internal_(type2, seq2_[l+1], seq2_[k-1]);
            return e;
        }
    }
    return e;
}

void
TurnerNearestNeighbor::
count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v)
{
    const auto type1 = complement_pair[seq2_[i]][seq2_[j]];
    const auto type2 = complement_pair[seq2_[l]][seq2_[k]];
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);

    if (ll==0) // stack
        count_stack_(type1, type2) += v;
    else if (ls==0) // bulge
    {
        if (use_count_bulge_at_least_)
        {
            if (ll<=30)
                for (auto k=ll; k>=1; --k) count_bulge_[k] += v;
            else
            {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
                for (auto k=30; k>=1; --k) count_bulge_[k] += v;
                count_lxc_[0] += v * log(ll / 30.);
#endif
            }
        }
        else
        {
            if (ll<=30)
                count_bulge_[ll] += v;
            else
            {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
                count_bulge_[30] += v;
                count_lxc_[0] += v * log(ll / 30.);
#endif
            }
        }

        if (ll==1) 
            count_stack_(type1, type2) += v;
        else
        {
            if (type1 > 2)
                count_terminalAU_[0] += v;
            if (type2 > 2)
                count_terminalAU_[0] += v;
        }
    }
    else // internal loop
    {
        if (ll==1 && ls==1) // 1x1 loop
            count_int11_(type1, type2, seq2_[i+1], seq2_[j-1]) += v;
        else if (l1==2 && l2==1) // 2x1 loop
            count_int21_(type2, type1, seq2_[l+1], seq2_[i+1], seq2_[k-1]) += v;
        else if (l1==1 && l2==2) // 1x2 loop
            count_int21_(type1, type2, seq2_[i+1], seq2_[l+1], seq2_[j-1]) += v;
        else if (ls==1) // 1xn loop
        {
            if (use_count_internal_at_least_)
            {
                if (ll+1 <= 30)
                    for (auto k=ll+1; k>=2; --k) count_internal_[k] += v;
                else
                {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
                    for (auto k=30; k>=2; --k) count_internal_[k] += v;
                    count_lxc_[0] += v * log((ll+1) / 30.);
#endif                    
                }
            }
            else
            {
                if (ll+1 <= 30)
                    count_internal_[ll+1] += v;
                else
                {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
                    count_internal_[30] += v;
                    count_lxc_[0] += v * log((ll+1) / 30.);
#endif
                }
            }
            
            if (score_max_ninio_[0] > (ll-ls) * score_ninio_[0])
                count_max_ninio_[0] += v;
            else
                count_ninio_[0] += v * (ll-ls);
            count_mismatch_internal_1n_(type1, seq2_[i+1], seq2_[j-1]) += v;
            count_mismatch_internal_1n_(type2, seq2_[l+1], seq2_[k-1]) += v;
        }
        else if (ls==2 && ll==2) // 2x2 loop
            count_int22_(type1, type2, seq2_[i+1], seq2_[k-1], seq2_[l+1], seq2_[j-1]) += v;
        else if (ls==2 && ll==3) // 2x3 loop
        {
            if (use_count_internal_at_least_)
                for (auto k=ls+ll; k>=2; --k) count_internal_[k] += v;
            else
                count_internal_[ls+ll] += v;
            count_ninio_[0] += v;
            count_mismatch_internal_23_(type1, seq2_[i+1], seq2_[j-1]) += v;
            count_mismatch_internal_23_(type2, seq2_[l+1], seq2_[k-1]) += v;
        }
        else // generic internal loop
        {
            if (use_count_internal_at_least_)
            {
                if (ls+ll <= 30)
                    for (auto k=ls+ll; k>=2; --k) count_internal_[k] += v;
                else
                {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
                    for (auto k=30; k>=2; --k) count_internal_[k] += v;
                    count_lxc_[0] += v * log((ls+ll) / 30.);
#endif
                }
            }
            else
            {
                if (ls+ll <= 30)
                    count_internal_[ls+ll] += v;
                else
                {
#if 0 // ignore very long unpaired regions that cannot be parsed in prediction
                    count_internal_[30] += v;
                    count_lxc_[0] += v * log((ls+ll) / 30.);
#endif
                }
            }
            
            if (score_max_ninio_[0] > (ll-ls) * score_ninio_[0])
                count_max_ninio_[0] += v;
            else
                count_ninio_[0] += v * (ll-ls);
            count_mismatch_internal_(type1, seq2_[i+1], seq2_[j-1]) += v;
            count_mismatch_internal_(type2, seq2_[l+1], seq2_[k-1]) += v;
        }
    }
}

auto
TurnerNearestNeighbor::
score_helix(size_t i, size_t j, size_t m) const -> ScoreType
{
    auto e = ScoreType(0.);
    for (auto k=1; k<m; k++)
    {
        const auto type1 = complement_pair[seq2_[i+(k-1)]][seq2_[j-(k-1)]];
        const auto type2 = complement_pair[seq2_[j-k]][seq2_[i+k]];
        e += score_stack_(type1, type2);
    }
    return e;
}

void
TurnerNearestNeighbor::
count_helix(size_t i, size_t j, size_t m, ScoreType v)
{
    auto e = ScoreType(0.);
    for (auto k=1; k<m; k++)
    {
        const auto type1 = complement_pair[seq2_[i+(k-1)]][seq2_[j-(k-1)]];
        const auto type2 = complement_pair[seq2_[j-k]][seq2_[i+k]];
        count_stack_(type1, type2) += v;
    }
}

auto
TurnerNearestNeighbor::
score_multi_loop(size_t i, size_t j) const -> ScoreType
{
    auto e = 0.;
    const auto type = complement_pair[seq2_[j]][seq2_[i]];
    e += score_mismatch_multi_(type, seq2_[j-1], seq2_[i+1]);
    if (type > 2) 
        e += score_terminalAU_[0];
    e += score_ml_intern_[0];
    e += score_ml_closing_[0];

    return e;
}

void
TurnerNearestNeighbor::
count_multi_loop(size_t i, size_t j, ScoreType v)
{
    const auto type = complement_pair[seq2_[j]][seq2_[i]];
    count_mismatch_multi_(type, seq2_[j-1], seq2_[i+1]) += v;
    if (type > 2) 
        count_terminalAU_[0] += v;
    count_ml_intern_[0] += v;
    count_ml_closing_[0] += v;
}

auto
TurnerNearestNeighbor::
score_multi_paired(size_t i, size_t j) const -> ScoreType
{
    const auto L = seq2_.size()-2;
    auto e = 0.;
    const auto type = complement_pair[seq2_[i]][seq2_[j]];
    if (i-1>=1 && j+1<=L)
        e += score_mismatch_multi_(type, seq2_[i-1], seq2_[j+1]);
    else if (i-1>=1)
        e += score_dangle5_(type, seq2_[i-1]);
    else if (j+1<=L)
        e += score_dangle3_(type, seq2_[j+1]);
    if (type > 2) 
        e += score_terminalAU_[0];
    e += score_ml_intern_[0];

    return e;
}

void
TurnerNearestNeighbor::
count_multi_paired(size_t i, size_t j, ScoreType v)
{
    const auto L = seq2_.size()-2;
    const auto type = complement_pair[seq2_[i]][seq2_[j]];
    if (i-1>=1 && j+1<=L)
        count_mismatch_multi_(type, seq2_[i-1], seq2_[j+1]) += v;
    else if (i-1>=1)
        count_dangle5_(type, seq2_[i-1]) += v;
    else if (j+1<=L)
        count_dangle3_(type, seq2_[j+1]) += v;
    if (type > 2) 
        count_terminalAU_[0] += v;
    count_ml_intern_[0] += v;
}

auto
TurnerNearestNeighbor::
score_multi_unpaired(size_t i, size_t j) const -> ScoreType
{
    return score_ml_base_[0] * (j-i+1);
}

void
TurnerNearestNeighbor::
count_multi_unpaired(size_t i, size_t j, ScoreType v)
{
    count_ml_base_[0] += (j-i+1);
}

auto
TurnerNearestNeighbor::
score_external_paired(size_t i, size_t j) const -> ScoreType
{
    const auto L = seq2_.size()-2;
    auto e = 0.;
    const auto type = complement_pair[seq2_[i]][seq2_[j]];
    if (i-1>=1 && j+1<=L)
        e += score_mismatch_external_(type, seq2_[i-1], seq2_[j+1]);
    else if (i-1>=1)
        e += score_dangle5_(type, seq2_[i-1]);
    else if (j+1<=L)
        e += score_dangle3_(type, seq2_[j+1]);
    if (type > 2) 
        e += score_terminalAU_[0];
    
    return e;
}

void
TurnerNearestNeighbor::
count_external_paired(size_t i, size_t j, ScoreType v)
{
    const auto L = seq2_.size()-2;
    const auto type = complement_pair[seq2_[i]][seq2_[j]];
    if (i-1>=1 && j+1<=L)
        count_mismatch_external_(type, seq2_[i-1], seq2_[j+1]) += v;
    else if (i-1>=1)
        count_dangle5_(type, seq2_[i-1]) += v;
    else if (j+1<=L)
        count_dangle3_(type, seq2_[j+1]) += v;
    if (type > 2) 
        count_terminalAU_[0] += v;
}
