#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <cassert>
#include "fold.h"
#include "parameter.h"

static
bool allow_paired(char x, char y)
{
    x = std::tolower(x);
    y = std::tolower(y);
    return (x=='a' && y=='u') || (x=='u' && y=='a') || 
        (x=='c' && y=='g') || (x=='g' && y=='c') ||
        (x=='g' && y=='u') || (x=='u' && y=='g');
}

static
auto parse_paren(const std::string& paren)
{
    std::vector<uint> bp(paren.size()+1, 0);
    std::stack<uint> st;
    for (auto i=0u; i!=paren.size(); ++i)
    {
        switch (paren[i])
        {
        case '(':
            st.push(i); break;
        case ')':
        {
            auto j=st.top();
            st.pop();
            bp[i+1] = j+1;
            bp[j+1] = i+1;
        }
        break;
        default: break;
        }
    }
    return bp;
}

static
auto make_constraint(const std::string& seq, std::string stru, u_int32_t max_bp, bool canonical_only=true)
{
    const auto L = seq.size();
    if (stru.size() < L)
        stru.append(std::string(L-stru.size(), '.'));
    else
        stru = stru.substr(0, L);
    auto bp = parse_paren(stru);

    if (canonical_only) // delete non-canonical base-pairs
        for (auto i=L; i>=1; i--)
            if (bp[i] > 0 && !::allow_paired(seq[i-1], seq[bp[i]-1]))
            {
                stru[i-1] = stru[bp[i]-1] = 'x';
                bp[i] = bp[bp[i]] = 0;
            }

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1, false));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1, false));
    //TriMatrix<bool> allow_paired(L+1, false, -1);
    //TriMatrix<bool> allow_unpaired(L+1, false, -1);
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired[i][i-1] = true; // the empty string is alway allowed to be unpaired
        allow_unpaired[i][i] = stru[i-1]=='.' || stru[i-1]=='x';
        bool bp_l = stru[i-1]=='.' || stru[i-1]=='<' || stru[i-1]=='|';
        for (auto j=i+1; j<=L; j++)
        {
            allow_paired[i][j] = j-i > max_bp;
            bool bp_r = stru[j-1]=='.' || stru[j-1]=='>' || stru[j-1]=='|';
            allow_paired[i][j] = allow_paired[i][j] && ((bp_l && bp_r) || bp[i]==j);
            if (canonical_only)
                allow_paired[i][j] = allow_paired[i][j] && ::allow_paired(seq[i-1], seq[j-1]);
            allow_unpaired[i][j] = allow_unpaired[i][j-1] && allow_unpaired[j][j];
        }
    }
    return std::make_pair(allow_paired, allow_unpaired);
}

static
auto make_penalty(size_t L, bool use_penalty, const std::string& ref, float pos_penalty, float neg_penalty)
{
    TriMatrix penalty(L+1, 0.0);
    if (use_penalty)
    {
        auto bp = parse_paren(ref);
        for (auto i=L; i>=1; i--)
            for (auto j=i+1; j<=L; j++)
                penalty[i][j] = bp[i] == j ? pos_penalty : neg_penalty;
    }
    return penalty;
}

template < typename P, typename S >
Fold<P, S>::
Fold(std::unique_ptr<P>&& p)
    :   param(std::move(p))
{

}

template < typename P, typename S >
bool
Fold<P, S>::
update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int32_t k)
{
    static const ScoreType NEG_INF2 = std::numeric_limits<ScoreType>::lowest()/1e10;
    if (max_v < new_v && NEG_INF2 < new_v)
    {
        max_v = new_v;
        max_t = {tt, k};
        return true;
    }
    return false;
}

template < typename P, typename S >
bool 
Fold<P, S>::
update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int8_t p, u_int8_t q)
{
    static const ScoreType NEG_INF2 = std::numeric_limits<ScoreType>::lowest()/1e10;
    if (max_v < new_v && NEG_INF2 < new_v)
    {
        max_v = new_v;
        max_t = {tt, std::make_pair(p, q)};
        return true;
    }
    return false;
}

template < typename P, typename S >
auto 
Fold<P, S>::
compute_viterbi(const std::string& seq, FoldOptions opts) -> ScoreType
{
    const auto seq2 = param->convert_sequence(seq);
    const auto L = seq.size();
    const ScoreType NEG_INF = std::numeric_limits<ScoreType>::lowest();
    Cv_.clear();  Cv_.resize(L+1, NEG_INF);
    Mv_.clear();  Mv_.resize(L+1, NEG_INF);
    M1v_.clear(); M1v_.resize(L+1, NEG_INF);
    Fv_.clear();  Fv_.resize(L+2, NEG_INF);
    Ct_.clear();  Ct_.resize(L+1);
    Mt_.clear();  Mt_.resize(L+1);
    M1t_.clear(); M1t_.resize(L+1);
    Ft_.clear();  Ft_.resize(L+2);

    const auto [allow_paired, allow_unpaired] = make_constraint(seq, opts.stru, opts.min_hairpin);
    const auto penalty = make_penalty(L, opts.use_penalty, opts.ref, opts.pos_penalty, opts.neg_penalty);

    std::vector<std::vector<u_int32_t>> split_point_c_l(L+1);
    std::vector<std::vector<u_int32_t>> split_point_c_r(L+1);
    std::vector<std::vector<u_int32_t>> split_point_m1_l(L+1);

    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (allow_paired[i][j])
            {
                bool suc = false;
                if (allow_unpaired[i+1][j-1])
                    suc |= update_max(Cv_[i][j], param->score_hairpin(seq2, i, j) + penalty[i][j], Ct_[i][j], TBType::C_HAIRPIN_LOOP);

                for (auto k=i+1; k<j && (k-1)-(i+1)+1<opts.max_internal; k++)
                {
                    if (!allow_unpaired[i+1][k-1]) break;
                    for (auto l=j-1; k<l && ((k-1)-(i+1)+1)+((j-1)-(l+1)+1)<opts.max_internal; l--)
                    {
                        if (!allow_unpaired[l+1][j-1]) break;
                        if (allow_paired[k][l])
                            suc |= update_max(Cv_[i][j], Cv_[k][l] + param->score_single_loop(seq2, i, j, k, l) + penalty[i][j], Ct_[i][j], TBType::C_INTERNAL_LOOP, k-i, j-l);
                    }
                }

                //for (auto u=i+2; u<=j-1; u++)
                for (auto u: split_point_m1_l[j-1])
                {
                    if (i+1>u-1) break;
                    suc |= update_max(Cv_[i][j], Mv_[i+1][u-1]+M1v_[u][j-1] + param->score_multi_loop(seq2, i, j) + penalty[i][j], Ct_[i][j], TBType::C_MULTI_LOOP, u);
                }
            
                if (suc)
                {
                    split_point_c_l[j].push_back(i);
                    split_point_c_r[i].push_back(j);
                }
            }

            /////////////////
            //for (auto u=i; u<j; u++)
            for (auto u: split_point_c_l[j])
            {
                if (i>u) break;
                if (allow_unpaired[i][u-1] /*&& allow_paired[u][j]*/) 
                {
                    auto t = param->score_multi_unpaired(seq2, u-1) * static_cast<float>(u-i);
                    auto s = param->score_multi_paired(seq2, u, j);
                    update_max(Mv_[i][j], Cv_[u][j] + s + t + penalty[u][j], Mt_[i][j], TBType::M_PAIRED, u);
                }
            }

            //for (auto u=i+1; u<=j; u++)
            for (auto u: split_point_c_l[j])
            {
                if (i>=u) break;
                //if (i<u /*&& allow_paired[u][j]*/)
                update_max(Mv_[i][j], Mv_[i][u-1]+Cv_[u][j] + param->score_multi_paired(seq2, u, j) + penalty[u][j], Mt_[i][j], TBType::M_BIFURCATION, u);
            }

            if (allow_unpaired[j][j])
                update_max(Mv_[i][j], Mv_[i][j-1] + param->score_multi_unpaired(seq2, j), Mt_[i][j], TBType::M_UNPAIRED);

            /////////////////
            bool suc = false;
            if (allow_paired[i][j])
                suc |= update_max(M1v_[i][j], Cv_[i][j] + param->score_multi_paired(seq2, i, j) + penalty[i][j], M1t_[i][j], TBType::M1_PAIRED);

            if (allow_unpaired[j][j])
                suc |= update_max(M1v_[i][j], M1v_[i][j-1] + param->score_multi_unpaired(seq2, j), M1t_[i][j], TBType::M1_UNPAIRED);

            if (suc) split_point_m1_l[j].push_back(i);
        }
    }

    update_max(Fv_[L+1], param->score_external_zero(seq2), Ft_[L+1], TBType::F_START);

    for (auto i=L; i>=1; i--)
    {
        if (allow_unpaired[i][i])
            update_max(Fv_[i], Fv_[i+1] + param->score_external_unpaired(seq2, i), Ft_[i], TBType::F_UNPAIRED);

        //for (auto k=i+1; k<=L; k++)
        for (auto k: split_point_c_r[i])
            //if (allow_paired[i][k])
            update_max(Fv_[i], Cv_[i][k]+Fv_[k+1] + param->score_external_paired(seq2, i, k) + penalty[i][k], Ft_[i], TBType::F_BIFURCATION, k);
    }

    return Fv_[1];
}

template < typename P, typename S >
auto
Fold<P, S>::
traceback_viterbi() -> std::vector<u_int32_t>
{
    const auto L = Ft_.size()-2;
    std::vector<u_int32_t> pair(L+1, 0);
    std::queue<std::tuple<TB, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Ft_[1], 1, L);

    while (!tb_queue.empty())
    {
        const auto [tb, i, j] = tb_queue.front();
        const auto [tb_type, kl] = tb;
        tb_queue.pop();

        switch (tb_type)
        {
            case TBType::C_HAIRPIN_LOOP: {
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(kl);
                const auto k = i+p;
                const auto l = j-q;
                assert(k < l);
                assert(pair[k] == 0);
                assert(pair[l] == 0);
                pair[k] = l;
                pair[l] = k;
                tb_queue.emplace(Ct_[k][l], k, l);
                break;
            }
            case TBType::C_MULTI_LOOP: {
                const auto u = std::get<0>(kl);
                tb_queue.emplace(Mt_[i+1][u-1], i+1, u-1);
                tb_queue.emplace(M1t_[u][j-1], u, j-1);
                break;
            }
            case TBType::M_PAIRED: {
                const auto k = std::get<0>(kl);
                assert(pair[k] == 0);
                assert(pair[j] == 0);
                pair[k] = j;
                pair[j] = k;
                tb_queue.emplace(Ct_[k][j], k, j);
                break;
            }
            case TBType::M_BIFURCATION: {
                const auto u = std::get<0>(kl);
                assert(pair[u] == 0);
                assert(pair[j] == 0);
                pair[u] = j;
                pair[j] = u;
                tb_queue.emplace(Mt_[i][u-1], i, u-1);
                tb_queue.emplace(Ct_[u][j], u, j);
                break;
            }
            case TBType::M_UNPAIRED: {
                tb_queue.emplace(Mt_[i][j-1], i, j-1);
                break;
            }    
            case TBType::M1_PAIRED: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                tb_queue.emplace(M1t_[i][j-1], i, j-1);
                break;
            }
            case TBType::F_START: {
                break;
            }
            case TBType::F_UNPAIRED: {
                tb_queue.emplace(Ft_[i+1], i+1, j);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(kl);
                assert(pair[i] == 0);
                assert(pair[k] == 0);
                pair[i] = k;
                pair[k] = i;
                tb_queue.emplace(Ct_[i][k], i, k);
                tb_queue.emplace(Ft_[k+1], k+1, j);
                break;
            }
        }
    }

    return pair;
}

template < typename P, typename S >
auto
Fold<P, S>::
traceback_viterbi(const std::string& seq, FoldOptions opts) -> typename P::ScoreType
{
    const auto seq2 = param->convert_sequence(seq);
    const auto L = Ft_.size()-2;
    const auto penalty = make_penalty(L, opts.use_penalty, opts.ref, opts.pos_penalty, opts.neg_penalty);
    std::queue<std::tuple<TB, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Ft_[1], 1, L);
    auto e = 0.;

    while (!tb_queue.empty())
    {
        const auto [tb, i, j] = tb_queue.front();
        const auto [tb_type, kl] = tb;
        tb_queue.pop();

        switch (tb_type)
        {
            case TBType::C_HAIRPIN_LOOP: {
                e += param->score_hairpin(seq2, i, j) + penalty[i][j];
                param->count_hairpin(seq2, i, j, 1.);
#if 0
                std::cout << "C_HAIRPIN_LOOP: " << i << ", " << j << ", " 
                    << param->template hairpin<typename P::ScoreType>(seq2, i, j).template item<float>() << std::endl;
#endif
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(kl);
                const auto k = i+p;
                const auto l = j-q;
                assert(k < l);
                e += param->score_single_loop(seq2, i, j, k, l) + penalty[i][j];
                param->count_single_loop(seq2, i, j, k, l, 1.);
#if 0
                std::cout << "C_INTERNAL_LOOP: " << i << ", " << j << ", " << k << ", " << l << ", " 
                    << param->template single_loop<typename P::ScoreType>(seq2, i, j, k, l).template item<float>() << std::endl;
#endif
                tb_queue.emplace(Ct_[k][l], k, l);
                break;
            }
            case TBType::C_MULTI_LOOP: {
                const auto u = std::get<0>(kl);
                e += param->score_multi_loop(seq2, i, j) + penalty[i][j];
                param->count_multi_loop(seq2, i, j, 1.);
#if 0
                std::cout << "C_MULTI_LOOP: " << i << ", " << j << ", " << k << ", "
                    << param->template multi_loop<typename P::ScoreType>(seq2, i, j).template item<float>() << std::endl;
#endif
                tb_queue.emplace(Mt_[i+1][u-1], i+1, u-1);
                tb_queue.emplace(M1t_[u][j-1], u, j-1);
                break;
            }
            case TBType::M_PAIRED: {
                const auto u = std::get<0>(kl);
                auto ee = param->score_multi_paired(seq2, u, j) + penalty[u][j];
                param->count_multi_paired(seq2, u, j, 1.);
                if (u-i > 0)
                {
                    ee += static_cast<float>(u-i) * param->score_multi_unpaired(seq2, u-1);
                    param->count_multi_unpaired(seq2, u-1, static_cast<float>(u-i));
                }
                e += ee; 
#if 0
                std::cout << "M_PAIRED: " << i << ", " << j << ", " << k << ", "
                    << ee.template item<float>() << std::endl;
#endif
                tb_queue.emplace(Ct_[u][j], u, j);
                break;
            }
            case TBType::M_BIFURCATION: {
                const auto u = std::get<0>(kl);
                e += param->score_multi_paired(seq2, u, j) + penalty[u][j];
                param->count_multi_paired(seq2, u, j, 1.);
#if 0
                std::cout << "M_BIRURCATION: " << i << ", " << j << ", " << k << ", "
                    << param->template multi_paired<typename P::ScoreType>(seq2, k+1, j).template item<float>() << std::endl;
#endif
                tb_queue.emplace(Mt_[i][u-1], i, u-1);
                tb_queue.emplace(Ct_[u][j], u, j);
                break;
            }
            case TBType::M_UNPAIRED: {
                e += param->score_multi_unpaired(seq2, j);
                param->count_multi_unpaired(seq2, j, 1.);
#if 0
                std::cout << "M_UNPAIRED: " << i << ", " << j << ", " 
                    << param->template multi_unpaired<typename P::ScoreType>(seq2, j).template item<float>() << std::endl;
#endif
                tb_queue.emplace(Mt_[i][j-1], i, j-1);
                break;
            }    
            case TBType::M1_PAIRED: {
                e += param->score_multi_paired(seq2, i, j) + penalty[i][j];
                param->count_multi_paired(seq2, i, j, 1.);
#if 0
                std::cout << "M1_PAIRED: " << i << ", " << j << ", " 
                    << param->template multi_paired<typename P::ScoreType>(seq2, i, j).template item<float>() << std::endl;
#endif
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                e += param->score_multi_unpaired(seq2, j);
                param->count_multi_unpaired(seq2, j, 1.);
#if 0
                std::cout << "M1_UNPAIRED: " << i << ", " << j << ", " 
                    << param->template multi_unpaired<typename P::ScoreType>(seq2, j).template item<float>() << std::endl;
#endif
                tb_queue.emplace(M1t_[i][j-1], i, j-1);
                break;
            }
            case TBType::F_START: {
                e += param->score_external_zero(seq2);
                param->count_external_zero(seq2, 1.);
#if 0
                std::cout << "F_START: " << i << ", " << j << ", " 
                    << param->template external_zero<typename P::ScoreType>(seq2).template item<float>() << std::endl;
#endif
                break;
            }
            case TBType::F_UNPAIRED: {
                e += param->score_external_unpaired(seq2, i);
                param->count_external_unpaired(seq2, i, 1.);
#if 0
                std::cout << "F_UNPAIRED: " << i << ", " << j << ", " 
                    << param->template external_unpaired<typename P::ScoreType>(seq2, i).template item<float>() << std::endl;
#endif
                tb_queue.emplace(Ft_[i+1], i+1, j);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(kl);
                e += param->score_external_paired(seq2, i, k) + penalty[i][k];
                param->count_external_paired(seq2, i, k, 1.);
#if 0
                std::cout << "F_BIFURCATION: " << i << ", " << j << ", " << k << ", " 
                    << param->template external_paired<typename P::ScoreType>(seq2, i, k).template item<float>() << std::endl;
#endif
                tb_queue.emplace(Ct_[i][k], i, k);
                tb_queue.emplace(Ft_[k+1], k+1, j);
                break;
            }
        }
    }

    return e;
}

// instantiation
#include "parameter.h"

template class Fold<MFE>;
template class Fold<PyMFE>;
