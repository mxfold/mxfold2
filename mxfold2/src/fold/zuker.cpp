#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <cassert>
#include "zuker.h"

template < typename P, typename S >
Zuker<P, S>::
Zuker(std::unique_ptr<P>&& p)
    : param_(std::move(p))
{

}

template < typename P, typename S >
bool
Zuker<P, S>::
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
Zuker<P, S>::
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
Zuker<P, S>::
compute_viterbi(const std::string& seq, Options opts) -> ScoreType
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
#ifdef HELIX_LENGTH
    Nv_.clear();  Nv_.resize(L+1, NEG_INF);
    Ev_.clear();  Ev_.resize(L+1, NEG_INF);
    Nt_.clear();  Nt_.resize(L+1);
    Et_.clear();  Et_.resize(L+1);
#endif

    const auto [allow_paired, allow_unpaired] = opts.make_constraint(seq);
    const auto [loss_paired, loss_unpaired, loss_const] = opts.make_penalty(L);

    std::vector<std::vector<u_int32_t>> split_point_c_l(L+1);
    std::vector<std::vector<u_int32_t>> split_point_c_r(L+1);
    std::vector<std::vector<u_int32_t>> split_point_m1_l(L+1);

    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
#ifdef HELIX_LENGTH
            if (allow_paired[i][j])
            {
                if (allow_unpaired[i+1][j-1]) 
                {
                    auto s = param_->score_hairpin(i, j) + loss_paired[i][j] + loss_unpaired[i+1][j-1];
                    update_max(Nv_[i][j], s, Nt_[i][j], TBType::N_HAIRPIN_LOOP);
                }

                for (auto k=i+1; k<j && (k-1)-(i+1)+1<=opts.max_internal; k++)
                {
                    if (!allow_unpaired[i+1][k-1]) break;
                    for (auto l=j-1; k<l && ((k-1)-(i+1)+1)+((j-1)-(l+1)+1)<=opts.max_internal; l--)
                    {
                        if (!allow_unpaired[l+1][j-1]) break;
                        if (((k-1)-(i+1)+1)+((j-1)-(l+1)+1)==0) continue; // nothoing to do here for stacking
                        if (allow_paired[k][l])
                        {
                            auto s = Cv_[k][l] + param_->score_single_loop(i, j, k, l) + loss_paired[i][j] + loss_unpaired[i+1][k-1] + loss_unpaired[l+1][j-1];
                            update_max(Nv_[i][j], s, Nt_[i][j], TBType::N_INTERNAL_LOOP, k-i, j-l);
                        }
                    }
                }

                //for (auto u=i+2; u<=j-1; u++)
                for (auto u: split_point_m1_l[j-1])
                {
                    if (i+1>u-1) break;
                    auto s = Mv_[i+1][u-1]+M1v_[u][j-1] + param_->score_multi_loop(i, j) + loss_paired[i][j];
                    update_max(Nv_[i][j], s, Nt_[i][j], TBType::N_MULTI_LOOP, u);
                }
            
                /////
                if (i+1 < j-1 && allow_paired[i+1][j-1]) 
                {
                    auto s = Ev_[i+1][j-1] + param_->score_single_loop(i, j, i+1, j-1) + loss_paired[i][j];
                    update_max(Ev_[i][j], s, Et_[i][j], TBType::E_HELIX);
                }
                update_max(Ev_[i][j], Nv_[i][j], Et_[i][j], TBType::E_TERMINAL);

                /////
                bool updated=false;
                if (opts.max_helix>0)
                {
                    // isolated base pair
                    updated |= update_max(Cv_[i][j], Nv_[i][j] + param_->score_helix(i, j, 1), Ct_[i][j], TBType::C_TERMINAL);
                    // helix (2~max_helix)
                    unsigned int m;
                    ScoreType lp = ScoreType(0.);
                    for (m=2; m<=opts.max_helix; m++)
                    {
                        if (i+(m-1)>=j-(m-1) || !allow_paired[i+(m-1)][j-(m-1)]) break;
                        lp += loss_paired[i+(m-2)][j-(m-2)];
                        auto s = Nv_[i+(m-1)][j-(m-1)] + param_->score_helix(i, j, m) + lp;
                        updated |= update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_HELIX, m);
                    }
                    if (m>opts.max_helix && i+(m-1)<j-(m-1) && allow_paired[i+(m-1)][j-(m-1)]) // long helix (max_helix+1~)
                    {
                        auto s = Ev_[i+(m-1)][j-(m-1)] + param_->score_helix(i, j, m) + lp;
                        updated |= update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_HELIX_E, m);
                    }
                }
                else
                {
                    updated |= update_max(Cv_[i][j], Ev_[i][j], Ct_[i][j], TBType::C_HELIX_E, 1);
                }

                if (updated)
                {
                    split_point_c_l[j].push_back(i);
                    split_point_c_r[i].push_back(j);
                }
            }
#else
            if (allow_paired[i][j])
            {
                bool updated=false;
                if (allow_unpaired[i+1][j-1]) 
                {
                    auto s = param_->score_hairpin(i, j) + loss_paired[i][j] + loss_unpaired[i+1][j-1];
                    updated |= update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_HAIRPIN_LOOP);
                }

                for (auto k=i+1; k<j && (k-1)-(i+1)+1<=opts.max_internal; k++)
                {
                    if (!allow_unpaired[i+1][k-1]) break;
                    for (auto l=j-1; k<l && ((k-1)-(i+1)+1)+((j-1)-(l+1)+1)<=opts.max_internal; l--)
                    {
                        if (!allow_unpaired[l+1][j-1]) break;
                        if (allow_paired[k][l])
                        {
                            auto s = Cv_[k][l] + param_->score_single_loop(i, j, k, l) + loss_paired[i][j] + loss_unpaired[i+1][k-1] + loss_unpaired[l+1][j-1];
                            updated |= update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_INTERNAL_LOOP, k-i, j-l);
                        }
                    }
                }

                //for (auto u=i+2; u<=j-1; u++)
                for (auto u: split_point_m1_l[j-1])
                {
                    if (i+1>u-1) break;
                    auto s = Mv_[i+1][u-1]+M1v_[u][j-1] + param_->score_multi_loop(i, j) + loss_paired[i][j];
                    updated |= update_max(Cv_[i][j], s, Ct_[i][j], TBType::C_MULTI_LOOP, u);
                }
            
                if (updated)
                {
                    split_point_c_l[j].push_back(i);
                    split_point_c_r[i].push_back(j);
                }
            }
#endif

            /////////////////
            //for (auto u=i; u<j; u++)
            for (auto u: split_point_c_l[j])
            {
                if (i>u) break;
                if (allow_unpaired[i][u-1] /*&& allow_paired[u][j]*/) 
                {
                    auto s = param_->score_multi_unpaired(i, u-1);
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
                auto s = Mv_[i][j-1] + param_->score_multi_unpaired(j, j) + loss_unpaired[j][j];
                update_max(Mv_[i][j], s, Mt_[i][j], TBType::M_UNPAIRED);
            }

            /////////////////
            bool updated=false;
            if (allow_paired[i][j])
            {
                auto s = Cv_[i][j] + param_->score_multi_paired(i, j);
                updated |= update_max(M1v_[i][j], s, M1t_[i][j], TBType::M1_PAIRED);
            }

            if (allow_unpaired[j][j])
            {
                auto s = M1v_[i][j-1] + param_->score_multi_unpaired(j, j) + loss_unpaired[j][j];
                updated |= update_max(M1v_[i][j], s, M1t_[i][j], TBType::M1_UNPAIRED);
            }

            if (updated) split_point_m1_l[j].push_back(i);
        }
    }

    update_max(Fv_[L+1], param_->score_external_zero(), Ft_[L+1], TBType::F_START);

    for (auto i=L; i>=1; i--)
    {
        if (allow_unpaired[i][i])
        {
            auto s = Fv_[i+1] + param_->score_external_unpaired(i, i) + loss_unpaired[i][i];
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
Zuker<P, S>::
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
#ifdef HELIX_LENGTH
            case TBType::N_HAIRPIN_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::N_INTERNAL_LOOP: {
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
            case TBType::N_MULTI_LOOP: {
                const auto u = std::get<0>(kl);
                tb_queue.emplace(Mt_[i+1][u-1], i+1, u-1);
                tb_queue.emplace(M1t_[u][j-1], u, j-1);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_HELIX: {
                tb_queue.emplace(Et_[i+1][j-1], i+1, j-1);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_TERMINAL: {
                tb_queue.emplace(Nt_[i][j], i, j);
                break;
            }
            case TBType::C_TERMINAL: {
                tb_queue.emplace(Nt_[i][j], i, j);
                break;
            }
            case TBType::C_HELIX: {
                const auto m = std::get<0>(kl);
                tb_queue.emplace(Nt_[i+(m-1)][j-(m-1)], i+(m-1), j-(m-1));
                for (auto k=2; k<=m; k++)
                {
                    assert(pair[i+(k-2)] == 0);
                    assert(pair[j-(k-2)] == 0);
                    pair[i+(k-2)] = j-(k-2);
                    pair[j-(k-2)] = i+(k-2);
                }
                break;
            }
            case TBType::C_HELIX_E: {
                const auto m = std::get<0>(kl);
                tb_queue.emplace(Et_[i+(m-1)][j-(m-1)], i+(m-1), j-(m-1));
                for (auto k=2; k<m; k++)
                {
                    assert(pair[i+(k-2)] == 0);
                    assert(pair[j-(k-2)] == 0);
                    pair[i+(k-2)] = j-(k-2);
                    pair[j-(k-2)] = i+(k-2);
                }
                break;
            }
#else
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
#endif
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
Zuker<P, S>::
traceback_viterbi(const std::string& seq, Options opts) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>
{
    const auto L = Ft_.size()-2;
    std::vector<u_int32_t> pair(L+1, 0);
    const auto [loss_paired, loss_unpaired, loss_const] = opts.make_penalty(L);
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
#ifdef HELIX_LENGTH
            case TBType::N_HAIRPIN_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                e += param_->score_hairpin(i, j) + loss_paired[i][j] + loss_unpaired[i+1][j-1];
                param_->count_hairpin(i, j, 1.);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::N_INTERNAL_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                const auto [p, q] = std::get<1>(kl);
                const auto k = i+p;
                const auto l = j-q;
                assert(k < l);
                e += param_->score_single_loop(i, j, k, l) + loss_paired[i][j] + loss_unpaired[i+1][k-1] + loss_unpaired[l+1][j-1];
                param_->count_single_loop(i, j, k, l, 1.);
                tb_queue.emplace(Ct_[k][l], k, l);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::N_MULTI_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                const auto u = std::get<0>(kl);
                e += param_->score_multi_loop(i, j) + loss_paired[i][j];
                param_->count_multi_loop(i, j, 1.);
                tb_queue.emplace(Mt_[i+1][u-1], i+1, u-1);
                tb_queue.emplace(M1t_[u][j-1], u, j-1);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_HELIX: {
                tb_queue.emplace(Et_[i+1][j-1], i+1, j-1);
                e += param_->score_single_loop(i, j, i+1, j-1) + loss_paired[i][j];
                param_->count_single_loop(i, j, i+1, j-1, 1.);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_TERMINAL: {
                tb_queue.emplace(Nt_[i][j], i, j);
                break;
            }
            case TBType::C_TERMINAL: {
                tb_queue.emplace(Nt_[i][j], i, j);
                e += param_->score_helix(i, j, 1);
                param_->count_helix(i, j, 1, 1.);
                break;
            }
            case TBType::C_HELIX: {
                const auto m = std::get<0>(kl);
                tb_queue.emplace(Nt_[i+(m-1)][j-(m-1)], i+(m-1), j-(m-1));
                ScoreType lp = 0.;
                for (auto k=2; k<=m; k++)
                {
                    assert(pair[i+(k-2)] == 0);
                    assert(pair[j-(k-2)] == 0);
                    pair[i+(k-2)] = j-(k-2);
                    pair[j-(k-2)] = i+(k-2);
                    lp += loss_paired[i+(k-2)][j-(k-2)];
                }
                e += param_->score_helix(i, j, m) + lp;
                param_->count_helix(i, j, m, 1.);
                break;
            }
            case TBType::C_HELIX_E: {
                const auto m = std::get<0>(kl);
                tb_queue.emplace(Et_[i+(m-1)][j-(m-1)], i+(m-1), j-(m-1));
                ScoreType lp = 0.;
                for (auto k=2; k<m; k++)
                {
                    assert(pair[i+(k-2)] == 0);
                    assert(pair[j-(k-2)] == 0);
                    pair[i+(k-2)] = j-(k-2);
                    pair[j-(k-2)] = i+(k-2);
                    lp += loss_paired[i+(k-2)][j-(k-2)];
                }
                e += param_->score_helix(i, j, m) + lp;
                param_->count_helix(i, j, m, 1.);
                break;
            }
#else
            case TBType::C_HAIRPIN_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                e += param_->score_hairpin(i, j) + loss_paired[i][j] + loss_unpaired[i+1][j-1];
                param_->count_hairpin(i, j, 1.);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                const auto [p, q] = std::get<1>(kl);
                const auto k = i+p;
                const auto l = j-q;
                assert(k < l);
                e += param_->score_single_loop(i, j, k, l) + loss_paired[i][j] + loss_unpaired[i+1][k-1] + loss_unpaired[l+1][j-1];
                param_->count_single_loop(i, j, k, l, 1.);
                tb_queue.emplace(Ct_[k][l], k, l);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::C_MULTI_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                const auto u = std::get<0>(kl);
                e += param_->score_multi_loop(i, j) + loss_paired[i][j];
                param_->count_multi_loop(i, j, 1.);
                tb_queue.emplace(Mt_[i+1][u-1], i+1, u-1);
                tb_queue.emplace(M1t_[u][j-1], u, j-1);
                pair[i] = j;
                pair[j] = i;
                break;
            }
#endif
            case TBType::M_PAIRED: {
                const auto u = std::get<0>(kl);
                e += param_->score_multi_paired(u, j) + param_->score_multi_unpaired(i, u-1) + loss_unpaired[i][u-1];
                param_->count_multi_paired(u, j, 1.);
                param_->count_multi_unpaired(i, u-1, 1.);
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
                e += param_->score_multi_unpaired(j, j) + loss_unpaired[j][j];
                param_->count_multi_unpaired(j, j, 1.);
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
                e += param_->score_multi_unpaired(j, j) + loss_unpaired[j][j];
                param_->count_multi_unpaired(j, j, 1.);
                tb_queue.emplace(M1t_[i][j-1], i, j-1);
                break;
            }
            case TBType::F_START: {
                e += param_->score_external_zero();
                param_->count_external_zero(1.);
                break;
            }
            case TBType::F_UNPAIRED: {
                e += param_->score_external_unpaired(i, i) + loss_unpaired[i][i];
                param_->count_external_unpaired(i, i, 1.);
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

    return std::make_pair(e + loss_const, pair);
}

// instantiation
#include "../param/turner.h"
#include "../param/positional.h"
#include "../param/mix.h"

template class Zuker<TurnerNearestNeighbor>;
template class Zuker<PositionalNearestNeighbor>;
template class Zuker<MixedNearestNeighbor>;