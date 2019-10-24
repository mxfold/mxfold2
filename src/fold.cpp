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

    for (auto i=L; i>=1; i--)
        if (bp[i] > 0) 
        {
            if ( (canonical_only && !::allow_paired(seq[i-1], seq[bp[i]-1])) || // delete non-canonical base-pairs
                    (bp[i] - i <= max_bp) ) // delete very short hairpin
            {
                stru[i-1] = stru[bp[i]-1] = 'x';
                bp[i] = bp[bp[i]] = 0;
            }
        }

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1, false));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1, false));
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
    TriMatrix p_paired(L+1, 0.0);
    std::vector<std::vector<float>> p_unpaired(L+1, std::vector<float>(L+1, 0.0));
    float p_const = 0;
    if (use_penalty)
    {
        auto bp = parse_paren(ref);
        for (auto i=L; i>=1; i--)
        {
            if (ref[i-1]=='.' || ref[i-1]=='x')
            {
                p_unpaired[i][i] = -pos_penalty;
                p_const += pos_penalty;
            }
            else
                p_unpaired[i][i] = neg_penalty;

            for (auto j=i+1; j<=L; j++)
            {
                p_unpaired[i][j] = p_unpaired[i][j-1] + p_unpaired[j][j];

                if (bp[i] == j)
                {
                    p_paired[i][j] = -pos_penalty;
                    p_const += pos_penalty;
                }
                else
                    p_paired[i][j] = neg_penalty;
            }
        }
    }
    return std::make_tuple(p_paired, p_unpaired, p_const);
}

template < typename P, typename S >
Fold<P, S>::
Fold(std::unique_ptr<P>&& p)
    :  param_(std::move(p))
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
    const auto [loss_paired, loss_unpaired, loss_const] = make_penalty(L, opts.use_penalty, opts.ref, opts.pos_penalty, opts.neg_penalty);

    std::vector<std::vector<u_int32_t>> split_point_c_l(L+1);
    std::vector<std::vector<u_int32_t>> split_point_c_r(L+1);
    std::vector<std::vector<u_int32_t>> split_point_m1_l(L+1);

    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (allow_paired[i][j])
            {
                bool suc1=false, suc2=false, suc3=false;
                if (allow_unpaired[i+1][j-1]) 
                {
                    auto s = param_->score_hairpin(i, j) + loss_paired[i][j] + loss_unpaired[i+1][j-1];
                    suc1 = update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_HAIRPIN_LOOP);
                }

                for (auto k=i+1; k<j && (k-1)-(i+1)+1<opts.max_internal; k++)
                {
                    if (!allow_unpaired[i+1][k-1]) break;
                    for (auto l=j-1; k<l && ((k-1)-(i+1)+1)+((j-1)-(l+1)+1)<opts.max_internal; l--)
                    {
                        if (!allow_unpaired[l+1][j-1]) break;
                        if (allow_paired[k][l])
                        {
                            auto s = Cv_[k][l] + param_->score_single_loop(i, j, k, l) + loss_paired[i][j] + loss_unpaired[i+1][k-1] + loss_unpaired[l+1][j-1];
                            suc2 = update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_INTERNAL_LOOP, k-i, j-l);
                        }
                    }
                }

                //for (auto u=i+2; u<=j-1; u++)
                for (auto u: split_point_m1_l[j-1])
                {
                    if (i+1>u-1) break;
                    auto s = Mv_[i+1][u-1]+M1v_[u][j-1] + param_->score_multi_loop(i, j) + loss_paired[i][j];
                    suc3 = update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_MULTI_LOOP, u);
                }
            
                if (suc1 || suc2 || suc3)
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
                    auto s = param_->score_multi_unpaired(u-1) * static_cast<float>(u-i);
                    auto t = param_->score_multi_paired(u, j);
                    auto r = Cv_[u][j] + s + t + loss_unpaired[i][u-1];
                    update_max(Mv_[i][j], r, Mt_[i][j], TBType::M_PAIRED, u);
                }
            }

            //for (auto u=i+1; u<=j; u++)
            for (auto u: split_point_c_l[j])
            {
                if (i>=u) break;
                //if (i<u /*&& allow_paired[u][j]*/)
                auto s = Mv_[i][u-1]+Cv_[u][j] + param_->score_multi_paired(u, j);
                update_max(Mv_[i][j], s, Mt_[i][j], TBType::M_BIFURCATION, u);
            }

            if (allow_unpaired[j][j])
            {
                auto s = Mv_[i][j-1] + param_->score_multi_unpaired(j) + loss_unpaired[j][j];
                update_max(Mv_[i][j], s, Mt_[i][j], TBType::M_UNPAIRED);
            }

            /////////////////
            bool suc1=false, suc2=false;
            if (allow_paired[i][j])
            {
                auto s = Cv_[i][j] + param_->score_multi_paired(i, j);
                suc1 = update_max(M1v_[i][j], s, M1t_[i][j], TBType::M1_PAIRED);
            }

            if (allow_unpaired[j][j])
            {
                auto s = M1v_[i][j-1] + param_->score_multi_unpaired(j) + loss_unpaired[j][j];
                suc2 = update_max(M1v_[i][j], s, M1t_[i][j], TBType::M1_UNPAIRED);
            }

            if (suc1 || suc2) split_point_m1_l[j].push_back(i);
        }
    }

    update_max(Fv_[L+1], param_->score_external_zero(), Ft_[L+1], TBType::F_START);

    for (auto i=L; i>=1; i--)
    {
        if (allow_unpaired[i][i])
        {
            auto s = Fv_[i+1] + param_->score_external_unpaired(i) + loss_unpaired[i][i];
            update_max(Fv_[i], s, Ft_[i], TBType::F_UNPAIRED);
        }

        //for (auto k=i+1; k<=L; k++)
        for (auto k: split_point_c_r[i])
        {
            //if (allow_paired[i][k])
            auto s = Cv_[i][k]+Fv_[k+1] + param_->score_external_paired(i, k);
            update_max(Fv_[i], s, Ft_[i], TBType::F_BIFURCATION, k);
        }
    }

    return Fv_[1] + loss_const;
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
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(kl);
                const auto k = i+p;
                const auto l = j-q;
                tb_queue.emplace(Ct_[k][l], k, l);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::C_MULTI_LOOP: {
                const auto u = std::get<0>(kl);
                tb_queue.emplace(Mt_[i+1][u-1], i+1, u-1);
                tb_queue.emplace(M1t_[u][j-1], u, j-1);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::M_PAIRED: {
                const auto k = std::get<0>(kl);
                tb_queue.emplace(Ct_[k][j], k, j);
                break;
            }
            case TBType::M_BIFURCATION: {
                const auto u = std::get<0>(kl);
                tb_queue.emplace(Mt_[i][u-1], i, u-1);
                tb_queue.emplace(Ct_[u][j], u, j);
                break;
            }
            case TBType::M_UNPAIRED: {
                tb_queue.emplace(Mt_[i][j-1], i, j-1);
                break;
            }    
            case TBType::M1_PAIRED: {
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
    const auto L = Ft_.size()-2;
    const auto [loss_paired, loss_unpaired, loss_const] = make_penalty(L, opts.use_penalty, opts.ref, opts.pos_penalty, opts.neg_penalty);
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
                e += param_->score_hairpin(i, j) + loss_paired[i][j] + loss_unpaired[i+1][j-1];
                param_->count_hairpin(i, j, 1.);
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(kl);
                const auto k = i+p;
                const auto l = j-q;
                assert(k < l);
                e += param_->score_single_loop(i, j, k, l) + loss_paired[i][j] + loss_unpaired[i+1][k-1] + loss_unpaired[l+1][j-1];
                param_->count_single_loop(i, j, k, l, 1.);
                tb_queue.emplace(Ct_[k][l], k, l);
                break;
            }
            case TBType::C_MULTI_LOOP: {
                const auto u = std::get<0>(kl);
                e += param_->score_multi_loop(i, j) + loss_paired[i][j];
                param_->count_multi_loop(i, j, 1.);
                tb_queue.emplace(Mt_[i+1][u-1], i+1, u-1);
                tb_queue.emplace(M1t_[u][j-1], u, j-1);
                break;
            }
            case TBType::M_PAIRED: {
                const auto u = std::get<0>(kl);
                auto ee = param_->score_multi_paired(u, j);
                param_->count_multi_paired(u, j, 1.);
                if (u-i > 0)
                {
                    ee += static_cast<float>(u-i) * param_->score_multi_unpaired(u-1) + loss_unpaired[i][u-1];
                    param_->count_multi_unpaired(u-1, static_cast<float>(u-i));
                }
                e += ee; 
                tb_queue.emplace(Ct_[u][j], u, j);
                break;
            }
            case TBType::M_BIFURCATION: {
                const auto u = std::get<0>(kl);
                e += param_->score_multi_paired(u, j);
                param_->count_multi_paired(u, j, 1.);
                tb_queue.emplace(Mt_[i][u-1], i, u-1);
                tb_queue.emplace(Ct_[u][j], u, j);
                break;
            }
            case TBType::M_UNPAIRED: {
                e += param_->score_multi_unpaired(j) + loss_unpaired[j][j];
                param_->count_multi_unpaired(j, 1.);
                tb_queue.emplace(Mt_[i][j-1], i, j-1);
                break;
            }    
            case TBType::M1_PAIRED: {
                e += param_->score_multi_paired(i, j);
                param_->count_multi_paired(i, j, 1.);
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                e += param_->score_multi_unpaired(j) + loss_unpaired[j][j];
                param_->count_multi_unpaired(j, 1.);
                tb_queue.emplace(M1t_[i][j-1], i, j-1);
                break;
            }
            case TBType::F_START: {
                e += param_->score_external_zero();
                param_->count_external_zero(1.);
                break;
            }
            case TBType::F_UNPAIRED: {
                e += param_->score_external_unpaired(i) + loss_unpaired[i][i];
                param_->count_external_unpaired(i, 1.);
                tb_queue.emplace(Ft_[i+1], i+1, j);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(kl);
                e += param_->score_external_paired(i, k);
                param_->count_external_paired(i, k, 1.);
                tb_queue.emplace(Ct_[i][k], i, k);
                tb_queue.emplace(Ft_[k+1], k+1, j);
                break;
            }
        }
    }

    return e + loss_const;
}

// instantiation
#include "parameter.h"

template class Fold<TurnerNearestNeighbor>;
template class Fold<PositionalNearestNeighbor>;