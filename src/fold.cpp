#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <torch/torch.h>
#include "fold.h"
#include "parameter.h"

bool allow_paired(char x, char y)
{
    x = std::tolower(x);
    y = std::tolower(y);
    return (x=='a' && y=='u') || (x=='u' && y=='a') || 
        (x=='c' && y=='g') || (x=='g' && y=='c') ||
        (x=='g' && y=='u') || (x=='u' && y=='g');
}

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

auto make_constraint(const std::string& seq, std::string stru, u_int32_t max_bp, bool canonical_only=true)
{
    const auto L = seq.size();
    if (stru.size() < L)
        stru.append(std::string(L-stru.size(), '.'));
    else
        stru = stru.substr(0, L);
    auto bp = parse_paren(stru);

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1));
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

template < typename T >
float compare(const T& a, const T& b)
{
    return a - b;
}

template <>
float compare(const torch::Tensor& a, const torch::Tensor& b)
{
    return a.item<float>() - b.item<float>();
}

template < typename T >
T NEG_INF()
{
    return std::numeric_limits<T>::lowest();
}

template <>
torch::Tensor NEG_INF()
{
    return torch::full({}, std::numeric_limits<float>::lowest(), torch::requires_grad(false));
}

template < typename T >
T ZERO()
{
    return 0.;
}

template <>
torch::Tensor ZERO()
{
    return torch::zeros({}, torch::dtype(torch::kFloat));
}

template < typename P, typename S >
Fold<P, S>::
Fold(std::unique_ptr<P>&& p, size_t min_hairpin_loop_length, size_t max_internal_loop_length)
    :   param(std::move(p)), 
        min_hairpin_loop_length_(min_hairpin_loop_length),
        max_internal_loop_length_(max_internal_loop_length)
{

}

template < typename P, typename S >
bool
Fold<P, S>::
update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int32_t k)
{
    if (::compare(max_v, new_v) < 0)
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
    if (::compare(max_v, new_v) < 0)
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
compute_viterbi(const std::string& seq, Fold<P, S>::options opts) -> ScoreType
{
    const auto seq2 = param->convert_sequence(seq);
    const auto L = seq.size();
    const ScoreType NEG_INF = ::NEG_INF<ScoreType>();
    Cv_.clear();  Cv_.resize(L+1, VI(L+1, NEG_INF));
    Mv_.clear();  Mv_.resize(L+1, VI(L+1, NEG_INF));
    M1v_.clear(); M1v_.resize(L+1, VI(L+1, NEG_INF));
    Fv_.clear();  Fv_.resize(L+1, NEG_INF);
    Ct_.clear();  Ct_.resize(L+1, VT(L+1));
    Mt_.clear();  Mt_.resize(L+1, VT(L+1));
    M1t_.clear(); M1t_.resize(L+1, VT(L+1));
    Ft_.clear();  Ft_.resize(L+1);

    const auto [allow_paired, allow_unpaired] = make_constraint(seq, opts.stru, min_hairpin_loop_length_);

    std::vector<std::vector<float>> penalty(L+1, std::vector<float>(L+1, 0.0));
    if (opts.use_penalty)
    {
        auto bp = parse_paren(opts.ref);
        for (auto i=L; i>=1; i--)
            for (auto j=i+1; j<=L; j++)
                penalty[i][j] = bp[i] == j ? opts.pos_penalty : opts.neg_penalty;
    }

    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (allow_paired[i][j])
            {
                if (allow_unpaired[i+1][j-1])
                    update_max(Cv_[i][j], param->template hairpin<ScoreType>(seq2, i, j) + penalty[i][j], Ct_[i][j], TBType::C_HAIRPIN_LOOP);

                for (auto k=i+1; (k-1)-(i+1)+1<max_internal_loop_length_ && k<j; k++)
                    if (allow_unpaired[i+1][k-1])
                        for (auto l=j-1; k<l && ((k-1)-(i+1)+1)+((j-1)-(l+1)+1)<max_internal_loop_length_; l--)
                            if (allow_paired[k][l] && allow_unpaired[l+1][j-1]) {
                                update_max(Cv_[i][j], Cv_[k][l] + param->template single_loop<ScoreType>(seq2, i, j, k, l) + penalty[i][j], Ct_[i][j], TBType::C_INTERNAL_LOOP, k-i, j-l);
                            }
                for (auto u=i+1; u+1<=j-1; u++)
                    update_max(Cv_[i][j], Mv_[i+1][u]+M1v_[u+1][j-1] + param->template multi_loop<ScoreType>(seq2, i, j) + penalty[i][j], Ct_[i][j], TBType::C_MULTI_LOOP, u);

            }

            /////////////////
            if (allow_paired[i][j])
                update_max(Mv_[i][j], Cv_[i][j] + param->template multi_paired<ScoreType>(seq2, i, j) + penalty[i][j], Mt_[i][j], TBType::M_PAIRED, i);

            ScoreType t = ::ZERO<ScoreType>();
            for (auto u=i; u+1<j; u++)
            {
                if (!allow_unpaired[u][u]) break;
                t += param->template multi_unpaired<ScoreType>(seq2, u);
                if (allow_paired[u+1][j])
                {
                    auto s = param->template multi_paired<ScoreType>(seq2, u+1, j) + t;
                    update_max(Mv_[i][j], Cv_[u+1][j] + s + penalty[u+1][j], Mt_[i][j], TBType::M_PAIRED, u+1);
                }
            }

            for (auto u=i; u+1<=j; u++)
                if (allow_paired[u+1][j])
                    update_max(Mv_[i][j], Mv_[i][u]+Cv_[u+1][j] + param->template multi_paired<ScoreType>(seq2, u+1, j) + penalty[u+1][j], Mt_[i][j], TBType::M_BIFURCATION, u);

            if (allow_unpaired[j][j])
                update_max(Mv_[i][j], Mv_[i][j-1] + param->template multi_unpaired<ScoreType>(seq2, j), Mt_[i][j], TBType::M_UNPAIRED);

            /////////////////
            if (allow_paired[i][j])
                update_max(M1v_[i][j], Cv_[i][j] + param->template multi_paired<ScoreType>(seq2, i, j) + penalty[i][j], M1t_[i][j], TBType::M1_PAIRED);

            if (allow_unpaired[j][j])
                update_max(M1v_[i][j], M1v_[i][j-1] + param->template multi_unpaired<ScoreType>(seq2, j), M1t_[i][j], TBType::M1_UNPAIRED);
        }
    }

    update_max(Fv_[L], param->template external_zero<ScoreType>(seq2), Ft_[L], TBType::F_ZERO);

    for (auto i=L-1; i>=1; i--)
    {
        if (allow_unpaired[i][i])
            update_max(Fv_[i], Fv_[i+1] + param->template external_unpaired<ScoreType>(seq2, i), Ft_[i], TBType::F_UNPAIRED);

        for (auto k=i+1; k+1<=L; k++)
            if (allow_paired[i][k])
                update_max(Fv_[i], Cv_[i][k]+Fv_[k+1] + param->template external_paired<ScoreType>(seq2, i, k) + penalty[i][k], Ft_[i], TBType::F_BIFURCATION, k);
    }

    if (allow_paired[1][L])
        update_max(Fv_[1], Cv_[1][L] + param->template external_paired<ScoreType>(seq2, 1, L) + penalty[1][L], Ft_[1], TBType::F_PAIRED);

    return Fv_[1];
}

template < typename P, typename S >
auto
Fold<P, S>::
traceback_viterbi() -> std::vector<u_int32_t>
{
    const auto L = Ft_.size()-1;
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
                const auto k = std::get<0>(kl);
                tb_queue.emplace(Mt_[i+1][k], i+1, k);
                tb_queue.emplace(M1t_[k+1][j-1], k+1, j-1);
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
                const auto k = std::get<0>(kl);
                assert(pair[k+1] == 0);
                assert(pair[j] == 0);
                pair[k+1] = j;
                pair[j] = k+1;
                tb_queue.emplace(Mt_[i][k], i, k);
                tb_queue.emplace(Ct_[k+1][j], k+1, j);
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
            case TBType::F_ZERO: {
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
            case TBType::F_PAIRED: {
                pair[i] = j;
                pair[j] = i;
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
        }
    }

    return pair;
}

template < typename P, typename S >
auto
Fold<P, S>::
traceback_viterbi(const std::string& seq) -> typename P::ScoreType
{
    const auto seq2 = param->convert_sequence(seq);
    const auto L = Ft_.size()-1;
    std::queue<std::tuple<TB, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Ft_[1], 1, L);
    auto e = ::ZERO<typename P::ScoreType>();

    while (!tb_queue.empty())
    {
        const auto [tb, i, j] = tb_queue.front();
        const auto [tb_type, kl] = tb;
        tb_queue.pop();

        switch (tb_type)
        {
            case TBType::C_HAIRPIN_LOOP: {
                e += param->template hairpin<typename P::ScoreType>(seq2, i, j);
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(kl);
                const auto k = i+p;
                const auto l = j-q;
                assert(k < l);
                e += param->template single_loop<typename P::ScoreType>(seq2, i, j, k, l);
                tb_queue.emplace(Ct_[k][l], k, l);
                break;
            }
            case TBType::C_MULTI_LOOP: {
                const auto k = std::get<0>(kl);
                e += param->template multi_loop<typename P::ScoreType>(seq2, i, j);
                tb_queue.emplace(Mt_[i+1][k], i+1, k);
                tb_queue.emplace(M1t_[k+1][j-1], k+1, j-1);
                break;
            }
            case TBType::M_PAIRED: {
                const auto k = std::get<0>(kl);
                if (k-i > 0)
                    e += static_cast<float>(k-i) * param->template multi_unpaired<typename P::ScoreType>(seq2, k);
                e += param->template multi_paired<typename P::ScoreType>(seq2, k, j);
                tb_queue.emplace(Ct_[k][j], k, j);
                break;
            }
            case TBType::M_BIFURCATION: {
                const auto k = std::get<0>(kl);
                e += param->template multi_paired<typename P::ScoreType>(seq2, k+1, j);
                tb_queue.emplace(Mt_[i][k], i, k);
                tb_queue.emplace(Ct_[k+1][j], k+1, j);
                break;
            }
            case TBType::M_UNPAIRED: {
                e += param->template multi_unpaired<typename P::ScoreType>(seq2, j);
                tb_queue.emplace(Mt_[i][j-1], i, j-1);
                break;
            }    
            case TBType::M1_PAIRED: {
                e += param->template multi_paired<typename P::ScoreType>(seq2, i, j);
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                e += param->template multi_unpaired<typename P::ScoreType>(seq2, j);
                tb_queue.emplace(M1t_[i][j-1], i, j-1);
                break;
            }
            case TBType::F_ZERO: {
                e += param->template external_zero<typename P::ScoreType>(seq2);
                break;
            }
            case TBType::F_UNPAIRED: {
                e += param->template external_unpaired<typename P::ScoreType>(seq2, i);
                tb_queue.emplace(Ft_[i+1], i+1, j);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(kl);
                e += param->template external_paired<typename P::ScoreType>(seq2, i, k);
                tb_queue.emplace(Ct_[i][k], i, k);
                tb_queue.emplace(Ft_[k+1], k+1, j);
                break;
            }
            case TBType::F_PAIRED: {
                e += param->template external_paired<typename P::ScoreType>(seq2, i, j);
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
        }
    }

    return e;
}

// instantiation
#include <torch/torch.h>
#include "parameter.h"

template class Fold<MFETorch>;
template class Fold<MFETorch, float>;
template class Fold<MFE>;


#if 0
auto nussinov(const std::string& seq)
{
    const auto L = seq.size();
    std::vector dp(L+1, std::vector(L+1, 0));
    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (j-i>3 && allow_paired(seq[i-1], seq[j-1]))
                dp[i][j] = std::max(dp[i][j], dp[i+1][j-1]+1);
            dp[i][j] = std::max(dp[i][j], dp[i+1][j]);
            dp[i][j] = std::max(dp[i][j], dp[i][j-1]);  
            for (auto k=i+1; k<j-1; k++)
                dp[i][j]  = std::max(dp[i][j], dp[i][k]+dp[k+1][j]);
        }
    }

    return dp[1][L];
}
#endif