#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <cassert>
#include <cmath>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "linfold.h"

template < typename P, typename S >
LinFold<P, S>::
LinFold(std::unique_ptr<P>&& p)
    : param_(std::move(p))
{

}

template < typename P, typename S >
bool
LinFold<P, S>::State::
update_max(S new_v, TBType tt, u_int32_t k)
{
    using ScoreType = S;
    static const ScoreType NEG_INF2 = std::numeric_limits<ScoreType>::lowest()/1e10;
    if (this->score < new_v && NEG_INF2 < new_v)
    {
        this->score = new_v;
        this->manner = tt;
        this->ptr = k;
        return true;
    }
    return false;
}

template < typename P, typename S >
bool 
LinFold<P, S>::State::
update_max(S new_v, TBType tt, u_int16_t p, u_int16_t q)
{
    using ScoreType = S;
    static const ScoreType NEG_INF2 = std::numeric_limits<ScoreType>::lowest()/1e10;
    if (this->score < new_v && NEG_INF2 < new_v)
    {
        this->score = new_v;
        this->manner = tt;
        this->ptr = std::make_pair(p, q);
        return true;
    }
    return false;
}

template < typename P, typename S >
auto
LinFold<P, S>::
beam_prune(std::unordered_map<u_int32_t, State>& states, u_int32_t beam_size) -> ScoreType
{
    static const ScoreType NEG_INF2 = std::numeric_limits<ScoreType>::lowest()/1e10;
    if (states.size() <= beam_size) return NEG_INF2;

    std::vector<std::pair<ScoreType, u_int32_t>> v;
    for (const auto& [i, st] : states)
    {
        auto k = i-1;
        ScoreType newscore = std::numeric_limits<ScoreType>::lowest();
        if (k==0)
            newscore = st.score;
        else if (Fv_[k].score > NEG_INF2)
            newscore = Fv_[k].score + st.score;
        v.emplace_back(newscore, i); 
    }

    std::sort(std::begin(v), std::end(v), [](const auto& x, const auto& y) { return x.first > y.first; });
    auto th = v[std::min<u_int32_t>(beam_size, v.size())-1].first;
    for (const auto& [s, i] : v)
        if (s < th)
            states.erase(i);

    return th;
}

template < typename P, typename S >
auto 
LinFold<P, S>::
compute_viterbi(const std::string& seq, const Options& opts) -> ScoreType
{
    // auto wtime = omp_get_wtime();
    auto beam_size = 100; // TODO: make this parameter
    const auto L = seq.size();
    const ScoreType NEG_INF = std::numeric_limits<ScoreType>::lowest();
    Hv_.clear();  Hv_.resize(L+1);
    Cv_.clear();  Cv_.resize(L+1);
    Mv_.clear();  Mv_.resize(L+1);
    M1v_.clear(); M1v_.resize(L+1);
    M2v_.clear(); M2v_.resize(L+1);
    Fv_.clear();  Fv_.resize(L+1);
#ifdef HELIX_LENGTH
    Nv_.clear();  Nv_.resize(L+1);
    Ev_.clear();  Ev_.resize(L+1);
#endif

    const auto [next_pair, allow_unpaired_range, allow_unpaired_position] = opts.make_constraint_lin(seq /*, "acgu"s */);

    Fv_[0].update_max(param_->score_external_zero(), TBType::F_START);
    if (L>0) Fv_[1].update_max(param_->score_external_unpaired(1, 1), TBType::F_UNPAIRED);
    if (L>1) Fv_[2].update_max(param_->score_external_unpaired(2, 2), TBType::F_UNPAIRED);

    for (auto j=1; j<=L; j++)
    {
        if (opts.stru[j]!=Options::UNPAIRED && opts.stru[j]!=Options::PAIRED_R)
        {
            // find a smallest hairpin loop candidate H(j, k)
            auto k = next_pair[seq[j-1]].size()>0 ? next_pair[seq[j-1]][j] : 0; // nearest k paired with j
            while (k>0 && k-j<=opts.min_hairpin)
                k = next_pair[seq[j-1]][k];
            if (opts.stru[j]<=L && opts.stru[j]>j) k=opts.stru[j]; // use direct base-pair constraint

            if (k>0 && allow_unpaired_range[j]>=k && opts.allow_paired(seq, j, k))
                Hv_[k][j].update_max(param_->score_hairpin(j, k), TBType::H_CLOSING);
        }

        // H: hairpin loops
        if (beam_size > 0) beam_prune(Hv_[j], beam_size);
        for (const auto& [i, st]: Hv_[j])
        {
#ifdef HELIX_LENGTH
            // N -> ( ... )
            auto newscore = st.score + opts.additional_paired_score(i, j);
            Nv_[j][i].update_max(newscore, TBType::N_HAIRPIN_LOOP);
#else
            // C -> ( ... )
            auto newscore = st.score + opts.additional_paired_score(i, j);
            Cv_[j][i].update_max(newscore, TBType::C_HAIRPIN_LOOP);
#endif

            // extend H(i, j) to H(i, k)
            auto k = next_pair[seq[i-1]].size()>0 ? next_pair[seq[i-1]][j] : 0;
            if (k>0 && allow_unpaired_range[i]>=k && opts.allow_paired(seq, i, k))
                Hv_[k][i].update_max(param_->score_hairpin(i, k), TBType::H_CLOSING);

        }
        if (j==1) continue; // TODO: really need this line?

        // M: multi loop candidates
        if (beam_size > 0) beam_prune(Mv_[j], beam_size);
        for (const auto& [i, st]: Mv_[j])
        {
#ifdef HELIX_LENGTH
            // N -> ( M )
            auto newscore = st.score + param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            Nv_[j][i].update_max(newscore, TBType::N_MULTI_LOOP);
#else
            // C -> ( M )
            auto newscore = st.score + param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            Cv_[j][i].update_max(newscore, TBType::C_MULTI_LOOP);
#endif

            // extend M(i, j) to M(i, k)
            auto k = next_pair[seq[i-1]].size()>0 ? next_pair[seq[i-1]][j] : 0;
            auto [l1, l2] = std::get<1>(st.ptr);
            if (k>0 && allow_unpaired_range[j]>=k && opts.allow_paired(seq, i, k))
            {
                auto newscore = st.score + param_->score_multi_unpaired(j+1, k-1);
                Mv_[k][i].update_max(newscore, TBType::M_CLOSING, l1, l2+k-j);
            }
        }

#ifdef HELIX_LENGTH
        // N: isolated closed loops
        if (beam_size > 0) beam_prune(Nv_[j], beam_size);
        for (const auto& [i, st]: Nv_[j])
        {
            // E -> N ; terminal of extended helix
            Ev_[j][i].update_max(st.score, TBType::E_TERMINAL);

            // C -> N ; isolated base-pair
            Cv_[j][i].update_max(st.score+param_->score_helix(i, j, 1), TBType::C_TERMINAL);

            // C -> ((( N ))) ; helix (< max_helix_length)
            ScoreType lp = ScoreType(0.);
            for (auto m=2; m<=opts.max_helix; m++)
            {
                if (i-(m-1)<1 || j+(m-1)>L || !opts.allow_paired(seq, i-(m-1), j+(m-1))) break;
                lp += opts.additional_paired_score(i-(m-1), j+(m-1));
                auto newscore = st.score + param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                Cv_[j+(m-1)][i-(m-1)].update_max(newscore, TBType::C_HELIX, m);
            }
        }

        // E: extended helices
        if (beam_size > 0) beam_prune(Ev_[j], beam_size);
        for (const auto& [i, st]: Ev_[j])
        {
            // E -> ( E ) ; extended helix longer than max_helix_length
            if (i-1>=1 && j+1<=L && opts.allow_paired(seq, i-1, j+1))
            {
                auto newscore = st.score + param_->score_single_loop(i-1, j+1, i, j) + opts.additional_paired_score(i-1, j+1);
                Ev_[j+1][i-1].update_max(newscore, TBType::E_HELIX);
            }

            // C -> ((( E ))) ; helix (= max_helix_length)
            ScoreType lp = ScoreType(0.);
            u_int32_t m;
            for (auto m=2; m<=opts.max_helix; m++)
            {
                if (i-(m-1)<1 || j+(m-1)>L || !opts.allow_paired(seq, i-(m-1), j+(m-1))) break;
                lp += opts.additional_paired_score(i-(m-1), j+(m-1));
            }
            if (m>opts.max_helix && i-(m-1)>=1 && j+(m-1)<=L && opts.allow_paired(seq, i-(m-1), j+(m-1)))
            {
                lp += opts.additional_paired_score(i-(m-1), j+(m-1));
                auto newscore = st.score + param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                Cv_[j+(m-1)][i-(m-1)].update_max(newscore, TBType::C_HELIX_E, m);
            }
        }
#endif

        // C: closed loops
        if (beam_size > 0) beam_prune(Cv_[j], beam_size);
        for (const auto& [i, st]: Cv_[j])
        {
            // M1 -> C
            auto newscore = st.score + param_->score_multi_paired(i, j);
            M1v_[j][i].update_max(newscore, TBType::M1_PAIRED);

            // M2 -> M1 C
            if (i-1>1 && !M1v_[i-1].empty()) 
            {
                auto M1_score = st.score + param_->score_multi_paired(i, j); // C -> M1
                auto it = M2v_[j].find(i);
                if (it==M2v_[j].end() || M1_score>it->second.score) // candidate list
                {
                    for (const auto& [l, st_l]: M1v_[i-1])
                        M2v_[j][l].update_max(M1_score+st_l.score, TBType::M2_BIFURCATION, i-1);
                }
            }
            
            // F -> F C
            if (i>=1)
            {
                auto newscore = Fv_[i-1].score + st.score + param_->score_external_paired(i, j);
                Fv_[j].update_max(newscore, TBType::F_BIFURCATION, i-1);
            }

#ifdef HELIX_LENGTH
            // N -> ( ... C ... )
            if (i>1 && j<L)
            {
                for (auto p=i-1; p>=1 && (i-1)-(p+1)+1<=opts.max_internal && allow_unpaired_range[p]>=i; --p) 
                {
                    auto q = next_pair[seq[p-1]].size()>0 ? next_pair[seq[p-1]][j] : 0;
                    while (q>0 && allow_unpaired_range[j]>=q && ((i-1)-(p+1)+1)+((q-1)-(j+1)+1)<=opts.max_internal)
                    {
                        if (opts.allow_paired(seq, p, q) && (i-p>1 || q-j>1))
                        {
                            auto newscore = st.score + param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            Nv_[q][p].update_max(newscore, TBType::N_INTERNAL_LOOP, i-p, q-j);
                        }
                        q = next_pair[seq[p-1]][q];
                    }
                }
            }
#else
            // C -> ( ... C ... )
            if (i>1 && j<L)
            {
                for (auto p=i-1; p>=1 && (i-1)-(p+1)+1<=opts.max_internal && allow_unpaired_range[p]>=i; --p) 
                {
                    auto q = next_pair[seq[p-1]].size()>0 ? next_pair[seq[p-1]][j] : 0;
                    while (q>0 && allow_unpaired_range[j]>=q && ((i-1)-(p+1)+1)+((q-1)-(j+1)+1)<=opts.max_internal)
                    {
                        if (opts.allow_paired(seq, p, q))
                        {
                            auto newscore = st.score + param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            Cv_[q][p].update_max(newscore, TBType::C_INTERNAL_LOOP, i-p, q-j);
                        }
                        q = next_pair[seq[p-1]][q];
                    }
                }
            }
#endif
        }

        // M2: multi loop candidates without unpaired bases
        if (beam_size>0) beam_prune(M2v_[j], beam_size);
        for (const auto& [i, st]: M2v_[j])
        {
            // M1 -> M2
            M1v_[j][i].update_max(st.score, TBType::M1_M2);

            // M -> ... M2 ...
            for (auto p=i-1; p>=1 && (i-1)-(p+1)+1<=opts.max_internal && allow_unpaired_range[p]>=i; --p) // TODO: is opts.max_internal OK?
            {
                auto q = next_pair[seq[p-1]].size()>0 ? next_pair[seq[p-1]][j] : 0;
                if (q>0 && allow_unpaired_range[j]>=q && opts.allow_paired(seq, p, q) /*&& ((i-1)-(p+1)+1)+((q-1)-(j+1)+1)<=opts.max_internal*/)
                {
                    auto newscore = param_->score_multi_unpaired(p+1, i-1) + param_->score_multi_unpaired(j+1, q-1) + st.score;
                    Mv_[q][p].update_max(newscore, TBType::M_CLOSING, i-p, q-j);
                }
            }
        }

        // M1: right-most multi loop candidates
        if (beam_size>0) beam_prune(M1v_[j], beam_size);
        for (const auto& [i, st]: M1v_[j])
        {
            // M1 -> M1 .
            if (j+1<L+1 && allow_unpaired_position[j+1])
            {
                auto newscore = st.score + param_->score_multi_unpaired(j+1, j+1);
                M1v_[j+1][i].update_max(newscore, TBType::M1_UNPAIRED);
            }
        }

        // F: external loops
        // F -> F .
        if (j+1<L+1 && allow_unpaired_position[j+1])
        {
            auto newscore = Fv_[j].score + param_->score_external_unpaired(j+1, j+1);
            Fv_[j+1].update_max(newscore, TBType::F_UNPAIRED);
        }
    }

    // std::cout << omp_get_wtime()-wtime << std::endl;
    return Fv_[L].score;
}

template < typename P, typename S >
auto
LinFold<P, S>::
traceback_viterbi() -> std::vector<u_int32_t>
{
    const auto L = Fv_.size()-1;
    std::vector<u_int32_t> pair(L+1, 0);
    std::queue<std::tuple<State, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Fv_[L], 1, L);

    while (!tb_queue.empty())
    {
        const auto [st, i, j] = tb_queue.front();
        tb_queue.pop();

        switch (st.manner)
        {
            case TBType::H_CLOSING:
                break;
#ifdef HELIX_LENGTH
            case TBType::N_HAIRPIN_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::N_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(st.ptr);
                const auto k = i+p;
                const auto l = j-q;
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                assert(k < l);
                tb_queue.emplace(Cv_[l][k], k, l);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::N_MULTI_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                tb_queue.emplace(Mv_[j][i], i, j);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_HELIX: {
                tb_queue.emplace(Ev_[j-1][i+1], i+1, j-1);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_TERMINAL: {
                tb_queue.emplace(Nv_[j][i], i, j);
                break;
            }
            case TBType::C_TERMINAL: {
                tb_queue.emplace(Nv_[j][i], i, j);
                break;
            }
            case TBType::C_HELIX: {
                const auto m = std::get<0>(st.ptr);
                tb_queue.emplace(Nv_[j-(m-1)][i+(m-1)], i+(m-1), j-(m-1));
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
                const auto m = std::get<0>(st.ptr);
                tb_queue.emplace(Ev_[j-(m-1)][i+(m-1)], i+(m-1), j-(m-1));
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
                const auto [p, q] = std::get<1>(st.ptr);
                const auto k = i+p;
                const auto l = j-q;
                tb_queue.emplace(Cv_[l][k], k, l);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::C_MULTI_LOOP: {
                tb_queue.emplace(Mv_[j][i], i, j);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
#endif
            case TBType::M_CLOSING: {
                const auto [p, q] = std::get<1>(st.ptr);
                const auto k = i+p;
                const auto l = j-q;
                tb_queue.emplace(M2v_[l][k], k, l);
                break;
            }
            case TBType::M2_BIFURCATION: {
                const auto k = std::get<0>(st.ptr);
                tb_queue.emplace(M1v_[k][i], i, k);
                tb_queue.emplace(Cv_[j][k+1], k+1, j);
                break;
            }
            case TBType::M1_M2: {
                tb_queue.emplace(M2v_[j][i], i, j);
                break;
            }            
            case TBType::M1_PAIRED: {
                tb_queue.emplace(Cv_[j][i], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                tb_queue.emplace(M1v_[j-1][i], i, j-1);
                break;
            }
            case TBType::NONE:
            case TBType::F_START: {
                break;
            }
            case TBType::F_UNPAIRED: {
                tb_queue.emplace(Fv_[j-1], i, j-1);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(st.ptr);
                tb_queue.emplace(Fv_[k], 1, k);
                tb_queue.emplace(Cv_[j][k+1], k+1, j);
                break;
            }
        }
    }

    return pair;
}

template < typename P, typename S >
auto
LinFold<P, S>::
traceback_viterbi(const std::string& seq, const Options& opts) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>
{
    const auto L = Fv_.size()-1;
    std::vector<u_int32_t> pair(L+1, 0);
    std::queue<std::tuple<State, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Fv_[L], 1, L);
    ScoreType e = 0.;

    while (!tb_queue.empty())
    {
        const auto [st, i, j] = tb_queue.front();
        tb_queue.pop();

        switch (st.manner)
        {
            case TBType::H_CLOSING: 
                break;
#ifdef HELIX_LENGTH
            case TBType::N_HAIRPIN_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                e += param_->score_hairpin(i, j) + opts.additional_paired_score(i, j);
                param_->count_hairpin(i, j, 1.);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::N_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(st.ptr);
                const auto k = i+p;
                const auto l = j-q;
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                assert(k < l);
                e += param_->score_single_loop(i, j, k, l) + opts.additional_paired_score(i, j);
                param_->count_single_loop(i, j, k, l, 1.);
                tb_queue.emplace(Cv_[l][k], k, l);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::N_MULTI_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                e += param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
                param_->count_multi_loop(i, j, 1.);
                tb_queue.emplace(Mv_[j][i], i, j);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_HELIX: {
                tb_queue.emplace(Ev_[j-1][i+1], i+1, j-1);
                e += param_->score_single_loop(i, j, i+1, j-1) + opts.additional_paired_score(i, j);
                param_->count_single_loop(i, j, i+1, j-1, 1.);
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::E_TERMINAL: {
                tb_queue.emplace(Nv_[j][i], i, j);
                break;
            }
            case TBType::C_TERMINAL: {
                tb_queue.emplace(Nv_[j][i], i, j);
                e += param_->score_helix(i, j, 1);
                param_->count_helix(i, j, 1, 1.);
                break;
            }
            case TBType::C_HELIX: {
                const auto m = std::get<0>(st.ptr);
                tb_queue.emplace(Nv_[j-(m-1)][i+(m-1)], i+(m-1), j-(m-1));
                ScoreType lp = 0.;
                for (auto k=2; k<=m; k++)
                {
                    assert(pair[i+(k-2)] == 0);
                    assert(pair[j-(k-2)] == 0);
                    pair[i+(k-2)] = j-(k-2);
                    pair[j-(k-2)] = i+(k-2);
                    lp += opts.additional_paired_score(i+(k-2), j-(k-2));
                }
                e += param_->score_helix(i, j, m) + lp;
                param_->count_helix(i, j, m, 1.);
                break;
            }
            case TBType::C_HELIX_E: {
                const auto m = std::get<0>(st.ptr);
                tb_queue.emplace(Ev_[j-(m-1)][i+(m-1)], i+(m-1), j-(m-1));
                ScoreType lp = 0.;
                for (auto k=2; k<m; k++)
                {
                    assert(pair[i+(k-2)] == 0);
                    assert(pair[j-(k-2)] == 0);
                    pair[i+(k-2)] = j-(k-2);
                    pair[j-(k-2)] = i+(k-2);
                    lp += opts.additional_paired_score(i+(k-2), j-(k-2));
                }
                e += param_->score_helix(i, j, m) + lp;
                param_->count_helix(i, j, m, 1.);
                break;
            }
#else
            case TBType::C_HAIRPIN_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                e += param_->score_hairpin(i, j) + opts.additional_paired_score(i, j);
                param_->count_hairpin(i, j, 1.);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(st.ptr);
                const auto k = i+p;
                const auto l = j-q;
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                assert(k < l);
                e += param_->score_single_loop(i, j, k, l) + opts.additional_paired_score(i, j);
                param_->count_single_loop(i, j, k, l, 1.);
                tb_queue.emplace(Cv_[l][k], k, l);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TBType::C_MULTI_LOOP: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                e += param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
                param_->count_multi_loop(i, j, 1.);
                tb_queue.emplace(Mv_[j][i], i, j);
                pair[i] = j;
                pair[j] = i;
                break;
            }
#endif
            case TBType::M_CLOSING: {
                const auto [p, q] = std::get<1>(st.ptr);
                const auto k = i+p;
                const auto l = j-q;
                e += param_->score_multi_unpaired(i+1, k-1) + param_->score_multi_unpaired(l+1, j-1);
                param_->count_multi_unpaired(i+1, k-1, 1.); 
                param_->count_multi_unpaired(l+1, j-1, 1.);
                tb_queue.emplace(M2v_[l][k], k, l);
                break;
            }
            case TBType::M2_BIFURCATION: {
                const auto k = std::get<0>(st.ptr);
                tb_queue.emplace(M1v_[k][i], i, k);
                e += param_->score_multi_paired(k+1, j);
                param_->count_multi_paired(k+1, j, 1.);
                tb_queue.emplace(Cv_[j][k+1], k+1, j);
                break;
            }
            case TBType::M1_M2: {
                tb_queue.emplace(M2v_[j][i], i, j);
                break;
            }            
            case TBType::M1_PAIRED: {
                e += param_->score_multi_paired(i, j);
                param_->count_multi_paired(i, j, 1.);
                tb_queue.emplace(Cv_[j][i], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                e += param_->score_multi_unpaired(j, j);
                param_->count_multi_unpaired(j, j, 1.);
                tb_queue.emplace(M1v_[j-1][i], i, j-1);
                break;
            }
            case TBType::NONE:
                break;
            case TBType::F_START: {
                e += param_->score_external_zero();
                param_->count_external_zero(1.);
                break;
            }
            case TBType::F_UNPAIRED: {
                e += param_->score_external_unpaired(j, j);
                param_->count_external_unpaired(j, j, 1.);
                tb_queue.emplace(Fv_[j-1], i, j-1);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(st.ptr);
                e += param_->score_external_paired(k+1, j);
                param_->count_external_paired(k+1, j, 1.);
                tb_queue.emplace(Fv_[k], 1, k);
                tb_queue.emplace(Cv_[j][k+1], k+1, j);
                break;
            }
        }
    }

    return std::make_pair(e, pair);
}

template <typename S>
auto
logsumexp(S x, S y)
{
    if (x > y) std::swap(x, y);
    return log1p(exp(x-y))+y;
}

// instantiation
#include "../param/turner.h"

template class LinFold<TurnerNearestNeighbor>;