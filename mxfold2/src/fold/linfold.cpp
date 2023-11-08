#include <iostream>
#include <algorithm>
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
auto
LinFold<P,S>::Options::
make_constraint(const std::string& seq, std::string alphabets /*="acgu"s*/, bool canonical_only /*=true*/) const
    -> std::tuple<std::vector<std::vector<u_int32_t>>, std::vector<u_int32_t>, std::vector<bool>>
{
    const auto L = seq.size();
    //std::vector<u_int32_t> stru(L+1, Options::ANY);
    //std::copy(std::begin(this->stru), std::end(this->stru), std::begin(stru));
    if (stru.size() == 0)
        stru.resize(L+1, Options::ANY);

    for (auto i=L; i>=1; i--)
    {
        if (stru[i] > 0 && stru[i] <= L) // paired
            if ( (canonical_only && !this->allow_paired(seq[i-1], seq[stru[i]-1])) || // delete non-canonical base-pairs
                    (stru[i] - i <= min_hairpin) ) // delete very short hairpin
                stru[i] = stru[stru[i]] = Options::UNPAIRED;
    }

    std::vector<bool> allow_unpaired_position(L+1, true);
    for (auto i=1; i<=L; i++)
        allow_unpaired_position[i] = stru[i]==Options::ANY || stru[i]==Options::UNPAIRED;

    std::vector<u_int32_t> allow_unpaired_range(L+1, 0);
    auto firstpair = L+1;
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired_range[i] = firstpair;
        if (!allow_unpaired_position[i])
            firstpair = i;
    }

    std::vector<std::vector<u_int32_t>> next_pair(256);
    for (auto nuc: alphabets)
    {
        next_pair[nuc].resize(L+1, 0);
        auto next = 0;
        for (auto j=L; j>=1; j--)
        {
            next_pair[nuc][j] = next;
            if (stru[j] != Options::UNPAIRED && this->allow_paired(seq[j-1], nuc))
                next = j;
        }
    }

    return { next_pair, allow_unpaired_range, allow_unpaired_position };
}

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
    const auto beam_size = opts.beam_size();
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

    const auto [next_pair, allow_unpaired_range, allow_unpaired_position] = opts.make_constraint(seq /*, "acgu"s */);

    Fv_[0].update_max(param_->score_external_zero(), TBType::F_START);
    if (L>0) Fv_[1].update_max(param_->score_external_unpaired(1, 1), TBType::F_UNPAIRED);
    if (L>1) Fv_[2].update_max(param_->score_external_unpaired(1, 2), TBType::F_UNPAIRED);

    for (auto j=1; j<=L; j++)
    {
        if (opts.stru.size()==0 || opts.stru[j]!=Options::UNPAIRED && opts.stru[j]!=Options::PAIRED_R)
        {
            // find a smallest hairpin loop candidate H(j, k)
            auto k = next_pair[seq[j-1]].size()>0 ? next_pair[seq[j-1]][j] : 0; // nearest k paired with j
            while (k>0 && k-j<=opts.min_hairpin)
                k = next_pair[seq[j-1]][k];
            if (opts.stru.size()>0 && opts.stru[j]<=L && opts.stru[j]>j) k=opts.stru[j]; // use direct base-pair constraint

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
            if (j+1<=L && allow_unpaired_position[j+1])
            {
                auto newscore = st.score + param_->score_multi_unpaired(j+1, j+1);
                M1v_[j+1][i].update_max(newscore, TBType::M1_UNPAIRED);
            }
        }

        // F: external loops
        // F -> F .
        if (j+1<=L && allow_unpaired_position[j+1])
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
    return x>y ? log1p(exp(y-x))+x : log1p(exp(x-y))+y;
}

template < typename P, typename S >
auto
LinFold<P, S>::
beam_prune(std::unordered_map<u_int32_t, AlphaBeta>& states, u_int32_t beam_size) -> ScoreType
{
    static const ScoreType NEG_INF2 = std::numeric_limits<ScoreType>::lowest()/1e10;
    if (states.size() <= beam_size) return NEG_INF2;

    std::vector<std::pair<ScoreType, u_int32_t>> v;
    for (const auto& [i, st] : states)
    {
        auto k = i-1;
        ScoreType newscore = std::numeric_limits<ScoreType>::lowest();
        if (k==0)
            newscore = st.alpha;
        else if (Fio_[k].alpha > NEG_INF2)
            newscore = Fio_[k].alpha + st.alpha;
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
compute_inside(const std::string& seq, const Options& opts) -> ScoreType
{
    // auto wtime = omp_get_wtime();
    const auto beam_size = opts.beam_size();
    const auto L = seq.size();
    const ScoreType NEG_INF = std::numeric_limits<ScoreType>::lowest();
    Hio_.clear();  Hio_.resize(L+1);
    Cio_.clear();  Cio_.resize(L+1);
    Mio_.clear();  Mio_.resize(L+1);
    M1io_.clear(); M1io_.resize(L+1);
    M2io_.clear(); M2io_.resize(L+1);
    Fio_.clear();  Fio_.resize(L+1);
#ifdef HELIX_LENGTH
    Nio_.clear();  Nio_.resize(L+1);
    Eio_.clear();  Eio_.resize(L+1);
#endif

    const auto [next_pair, allow_unpaired_range, allow_unpaired_position] = opts.make_constraint(seq /*, "acgu"s */);

    Fio_[0].alpha = logsumexp(Fio_[0].alpha, param_->score_external_zero());
    if (L>0) Fio_[1].alpha = logsumexp(Fio_[1].alpha, param_->score_external_unpaired(1, 1));
    if (L>1) Fio_[2].alpha = logsumexp(Fio_[2].alpha, param_->score_external_unpaired(1, 2));

    for (auto j=1; j<=L; j++)
    {
        if (opts.stru.size()==0 || opts.stru[j]!=Options::UNPAIRED && opts.stru[j]!=Options::PAIRED_R)
        {
            // find a smallest hairpin loop candidate H(j, k)
            auto k = next_pair[seq[j-1]].size()>0 ? next_pair[seq[j-1]][j] : 0; // nearest k paired with j
            while (k>0 && k-j<=opts.min_hairpin)
                k = next_pair[seq[j-1]][k];
            if (opts.stru.size()>0 && opts.stru[j]<=L && opts.stru[j]>j) k=opts.stru[j]; // use direct base-pair constraint

            if (k>0 && allow_unpaired_range[j]>=k && opts.allow_paired(seq, j, k))
                Hio_[k][j].alpha = logsumexp(Hio_[k][j].alpha, param_->score_hairpin(j, k)); // TBType::H_CLOSING
        }

        // H: hairpin loops
        if (beam_size > 0) beam_prune(Hio_[j], beam_size);
        for (const auto& [i, st]: Hio_[j])
        {
#ifdef HELIX_LENGTH
            // N -> ( ... )
            auto newscore = st.alpha + opts.additional_paired_score(i, j);
            Nio_[j][i].alpha = logsumexp(Nio_[j][i].alpha, newscore); // TBType::N_HAIRPIN_LOOP
#else
            // C -> ( ... )
            auto newscore = st.alpha + opts.additional_paired_score(i, j);
            Cio_[j][i].alpha = logsumexp(Cio_[j][i].alpha, newscore); // TBType::C_HAIRPIN_LOOP
#endif

            // extend H(i, j) to H(i, k)
            auto k = next_pair[seq[i-1]].size()>0 ? next_pair[seq[i-1]][j] : 0;
            if (k>0 && allow_unpaired_range[i]>=k && opts.allow_paired(seq, i, k))
                Hio_[k][i].alpha = logsumexp(Hio_[k][i].alpha, param_->score_hairpin(i, k)); // TBType::H_CLOSING

        }
        if (j==1) continue; // TODO: really need this line?

        // M: multi loop candidates
        if (beam_size > 0) beam_prune(Mio_[j], beam_size);
        for (const auto& [i, st]: Mio_[j])
        {
#ifdef HELIX_LENGTH
            // N -> ( M )
            auto newscore = st.alpha + param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            Nio_[j][i].alpha = logsumexp(Nio_[j][i].alpha, newscore); // TBType::N_MULTI_LOOP
#else
            // C -> ( M )
            auto newscore = st.alpha + param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            Cio_[j][i].alpha = logsumexp(Cio_[j][i].alpha, newscore); // TBType::C_MULTI_LOOP
#endif

            // extend M(i, j) to M(i, k)
            auto k = next_pair[seq[i-1]].size()>0 ? next_pair[seq[i-1]][j] : 0;
            // auto [l1, l2] = std::get<1>(st.ptr);
            if (k>0 && allow_unpaired_range[j]>=k && opts.allow_paired(seq, i, k))
            {
                auto newscore = st.alpha + param_->score_multi_unpaired(j+1, k-1);
                Mio_[k][i].alpha = logsumexp(Mio_[k][i].alpha, newscore); // TBType::M_CLOSING
            }
        }

#ifdef HELIX_LENGTH
        // N: isolated closed loops
        if (beam_size > 0) beam_prune(Nio_[j], beam_size);
        for (const auto& [i, st]: Nio_[j])
        {
            // E -> N ; terminal of extended helix
            Eio_[j][i].alpha = logsumexp(Eio_[j][i].alpha, st.alpha); // TBType::E_TERMINAL

            // C -> N ; isolated base-pair
            Cio_[j][i].alpha = logsumexp(Cio_[j][i].alpha, st.alpha+param_->score_helix(i, j, 1)); // TBType::C_TERMINAL

            // C -> ((( N ))) ; helix (< max_helix_length)
            ScoreType lp = ScoreType(0.);
            for (auto m=2; m<=opts.max_helix; m++)
            {
                if (i-(m-1)<1 || j+(m-1)>L || !opts.allow_paired(seq, i-(m-1), j+(m-1))) break;
                lp += opts.additional_paired_score(i-(m-1), j+(m-1));
                auto newscore = st.alpha + param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                Cio_[j+(m-1)][i-(m-1)].alpha = logsumexp(Cio_[j+(m-1)][i-(m-1)].alpha, newscore); // TBType::C_HELIX
            }
        }

        // E: extended helices
        if (beam_size > 0) beam_prune(Eio_[j], beam_size);
        for (const auto& [i, st]: Eio_[j])
        {
            // E -> ( E ) ; extended helix longer than max_helix_length
            if (i-1>=1 && j+1<=L && opts.allow_paired(seq, i-1, j+1))
            {
                auto newscore = st.alpha + param_->score_single_loop(i-1, j+1, i, j) + opts.additional_paired_score(i-1, j+1);
                Eio_[j+1][i-1].alpha = logsumexp(Eio_[j+1][i-1].alpha, newscore); // TBType::E_HELIX
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
                auto newscore = st.alpha + param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                Cio_[j+(m-1)][i-(m-1)].alpha = logsumexp(Cio_[j+(m-1)][i-(m-1)].alpha, newscore); // TBType::C_HELIX_E
            }
        }
#endif

        // C: closed loops
        if (beam_size > 0) beam_prune(Cio_[j], beam_size);
        for (const auto& [i, st]: Cio_[j])
        {
            // M1 -> C
            auto newscore = st.alpha + param_->score_multi_paired(i, j);
            M1io_[j][i].alpha = logsumexp(M1io_[j][i].alpha, newscore); // TBType::M1_PAIRED

            // M2 -> M1 C
            if (i-1>1 && !M1io_[i-1].empty()) 
            {
                auto M1_score = st.alpha + param_->score_multi_paired(i, j); // C -> M1
                for (const auto& [l, st_l]: M1io_[i-1])
                    M2io_[j][l].alpha = logsumexp(M2io_[j][l].alpha, M1_score+st_l.alpha); // TBType::M2_BIFURCATION
            }
            
            // F -> F C
            if (i>=1)
            {
                auto newscore = Fio_[i-1].alpha + st.alpha + param_->score_external_paired(i, j);
                Fio_[j].alpha = logsumexp(Fio_[j].alpha, newscore); // TBType::F_BIFURCATION;
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
                            auto newscore = st.alpha + param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            Nio_[q][p].alpha = logsumexp(Nio_[q][p].alpha, newscore); // TBType::N_INTERNAL_LOOP
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
                            auto newscore = st.alpha + param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            Cio_[q][p].alpha = logsumexp(Cio_[q][p].alpha, newscore); // TBType::C_INTERNAL_LOOP
                        }
                        q = next_pair[seq[p-1]][q];
                    }
                }
            }
#endif
        }

        // M2: multi loop candidates without unpaired bases
        if (beam_size>0) beam_prune(M2io_[j], beam_size);
        for (const auto& [i, st]: M2io_[j])
        {
            // M1 -> M2
            M1io_[j][i].alpha = logsumexp(M1io_[j][i].alpha, st.alpha); // TBType::M1_M2

            // M -> ... M2 ...
            for (auto p=i-1; p>=1 && (i-1)-(p+1)+1<=opts.max_internal && allow_unpaired_range[p]>=i; --p) // TODO: is opts.max_internal OK?
            {
                auto q = next_pair[seq[p-1]].size()>0 ? next_pair[seq[p-1]][j] : 0;
                if (q>0 && allow_unpaired_range[j]>=q && opts.allow_paired(seq, p, q) /*&& ((i-1)-(p+1)+1)+((q-1)-(j+1)+1)<=opts.max_internal*/)
                {
                    auto newscore = param_->score_multi_unpaired(p+1, i-1) + param_->score_multi_unpaired(j+1, q-1) + st.alpha;
                    Mio_[q][p].alpha = logsumexp(Mio_[q][p].alpha, newscore); // TBType::M_CLOSING
                }
            }
        }

        // M1: right-most multi loop candidates
        if (beam_size>0) beam_prune(M1io_[j], beam_size);
        for (const auto& [i, st]: M1io_[j])
        {
            // M1 -> M1 .
            if (j+1<=L && allow_unpaired_position[j+1])
            {
                auto newscore = st.alpha + param_->score_multi_unpaired(j+1, j+1);
                M1io_[j+1][i].alpha = logsumexp(M1io_[j+1][i].alpha, newscore); // TBType::M1_UNPAIRED
            }
        }

        // F: external loops
        // F -> F .
        if (j+1<=L && allow_unpaired_position[j+1])
        {
            auto newscore = Fio_[j].alpha + param_->score_external_unpaired(j+1, j+1);
            Fio_[j+1].alpha = logsumexp(Fio_[j+1].alpha, newscore); // TBType::F_UNPAIRED
        }
    }

    // std::cout << omp_get_wtime()-wtime << std::endl;
    return Fio_[L].alpha;
}

template < typename P, typename S >
void 
LinFold<P, S>::
compute_outside(const std::string& seq, const Options& opts)
{
    const auto L = seq.size();
    const ScoreType NEG_INF = std::numeric_limits<ScoreType>::lowest();

    const auto [next_pair, allow_unpaired_range, allow_unpaired_position] = opts.make_constraint(seq /*, "acgu"s */); // TODO: reuse these values from inside computation

    Fio_[L].beta = ScoreType(0.);

    for (auto j=L; j>1; j--)
    {
        // F: external loops
        // F -> F .
        if (j+1<=L && allow_unpaired_position[j+1])
        {
            auto newscore = param_->score_external_unpaired(j+1, j+1);
            Fio_[j].beta = logsumexp(Fio_[j].beta, Fio_[j+1].beta + newscore); // TBType::F_UNPAIRED
        }

        // M1: right-most multi loop candidates
        for (auto& [i, st]: M1io_[j])
        {
            // M1 -> M1 .
            if (j+1<=L && allow_unpaired_position[j+1])
            {
                auto newscore = param_->score_multi_unpaired(j+1, j+1);
                st.beta = logsumexp(st.beta, M1io_[j+1][i].beta + newscore); // TBType::M1_UNPAIRED
            }
        }

        // M2: multi loop candidates without unpaired bases
        for (auto& [i, st]: M2io_[j])
        {
            // M1 -> M2
            st.beta = logsumexp(st.beta, M1io_[j][i].beta); // TBType::M1_M2

            // M -> ... M2 ...
            for (auto p=i-1; p>=1 && (i-1)-(p+1)+1<=opts.max_internal && allow_unpaired_range[p]>=i; --p) // TODO: is opts.max_internal OK?
            {
                auto q = next_pair[seq[p-1]].size()>0 ? next_pair[seq[p-1]][j] : 0;
                if (q>0 && allow_unpaired_range[j]>=q && opts.allow_paired(seq, p, q) /*&& ((i-1)-(p+1)+1)+((q-1)-(j+1)+1)<=opts.max_internal*/)
                {
                    auto newscore = param_->score_multi_unpaired(p+1, i-1) + param_->score_multi_unpaired(j+1, q-1);
                    st.beta = logsumexp(st.beta, Mio_[q][p].beta + newscore); // TBType::M_CLOSING
                }
            }
        }

        // C: closed loops
        for (auto& [i, st]: Cio_[j])
        {
            // M1 -> C
            auto newscore = param_->score_multi_paired(i, j);
            st.beta = logsumexp(st.beta, M1io_[j][i].beta + newscore); // TBType::M1_PAIRED

            // M2 -> M1 C
            if (i-1>1 && !M1io_[i-1].empty()) 
            {
                auto M1_score = param_->score_multi_paired(i, j); // C -> M1
                for (auto& [l, st_l]: M1io_[i-1])
                {
                    st.beta = logsumexp(st.beta, M2io_[j][l].beta + M1_score+st_l.alpha); // TBType::M2_BIFURCATION
                    st_l.beta = logsumexp(st_l.beta, M2io_[j][l].beta + M1_score+st.alpha); // TBType::M2_BIFURCATION
                }
            }
            
            // F -> F C
            if (i>=1)
            {
                auto newscore = param_->score_external_paired(i, j);
                Fio_[i-1].beta = logsumexp(Fio_[i-1].beta, Fio_[j].beta + st.alpha + newscore); // TBType::F_BIFURCATION
                st.beta = logsumexp(st.beta, Fio_[j].beta + Fio_[i-1].alpha + newscore); // TBType::F_BIFURCATION
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
                            auto newscore = param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            st.beta = logsumexp(st.beta, Nio_[q][p].beta + newscore); // TBType::N_INTERNAL_LOOP
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
                            auto newscore = param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            st.beta = logsumexp(st.beta, Cio_[q][p].beta + newscore); // TBType::C_INTERNAL_LOOP
                        }
                        q = next_pair[seq[p-1]][q];
                    }
                }
            }
#endif
        }

#ifdef HELIX_LENGTH
        // E: extended helices
        for (auto& [i, st]: Eio_[j])
        {
            // E -> ( E ) ; extended helix longer than max_helix_length
            if (i-1>=1 && j+1<=L && opts.allow_paired(seq, i-1, j+1))
            {
                auto newscore = param_->score_single_loop(i-1, j+1, i, j) + opts.additional_paired_score(i-1, j+1);
                st.beta = logsumexp(st.beta, Eio_[j+1][i-1].beta + newscore); // TBType::E_HELIX
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
                auto newscore = param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                st.beta = logsumexp(st.beta, Cio_[j+(m-1)][i-(m-1)].beta + newscore); // TBType::C_HELIX_E
            }
        }

        // N: isolated closed loops
        for (auto& [i, st]: Nio_[j])
        {
            // E -> N ; terminal of extended helix
            st.beta = logsumexp(st.beta, Eio_[j][i].beta); // TBType::E_TERMINAL

            // C -> N ; isolated base-pair
            st.beta = logsumexp(st.beta, Cio_[j][i].beta + param_->score_helix(i, j, 1)); // TBType::C_TERMINAL

            // C -> ((( N ))) ; helix (< max_helix_length)
            ScoreType lp = ScoreType(0.);
            for (auto m=2; m<=opts.max_helix; m++)
            {
                if (i-(m-1)<1 || j+(m-1)>L || !opts.allow_paired(seq, i-(m-1), j+(m-1))) break;
                lp += opts.additional_paired_score(i-(m-1), j+(m-1));
                auto newscore = param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                st.beta = logsumexp(st.beta, Cio_[j+(m-1)][i-(m-1)].beta + newscore); // TBType::C_HELIX
            }
        }
#endif
        // M: multi loop candidates
        for (auto& [i, st]: Mio_[j])
        {
#ifdef HELIX_LENGTH
            // N -> ( M )
            auto newscore = param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            st.beta = logsumexp(st.beta, Nio_[j][i].beta + newscore); // TBType::N_MULTI_LOOP
#else
            // C -> ( M )
            auto newscore = param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            st.beta = logsumexp(st.beta, Cio_[j][i].beta + newscore); // TBType::C_MULTI_LOOP
#endif

            // extend M(i, j) to M(i, k)
            auto k = next_pair[seq[i-1]].size()>0 ? next_pair[seq[i-1]][j] : 0;
            // auto [l1, l2] = std::get<1>(st.ptr);
            if (k>0 && allow_unpaired_range[j]>=k && opts.allow_paired(seq, i, k))
            {
                auto newscore = param_->score_multi_unpaired(j+1, k-1);
                st.beta = logsumexp(st.beta, Mio_[k][i].beta + newscore); // TBType::M_CLOSING
            }
        }
    }
}

template < typename P, typename S >
auto 
LinFold<P, S>::
compute_basepairing_probabilities(const std::string& seq, const Options& opts) -> std::vector<std::vector<std::pair<u_int32_t, float>>>
{
    // auto wtime = omp_get_wtime();
    const auto L = seq.size();
    const ScoreType NEG_INF = std::numeric_limits<ScoreType>::lowest();
    const auto log_partition_coefficient = Fio_[L].alpha;
    std::vector<std::unordered_map<u_int32_t, float>> bpp(L+1);

    const auto [next_pair, allow_unpaired_range, allow_unpaired_position] = opts.make_constraint(seq /*, "acgu"s */); // TODO: reuse these values from inside computation

    for (auto j=1; j<=L; j++)
    {
        // H: hairpin loops
        for (const auto& [i, st]: Hio_[j])
        {
#ifdef HELIX_LENGTH
            // N -> ( ... )
            auto newscore = st.alpha + opts.additional_paired_score(i, j);
            assert(newscore + Nio_[j][i].beta <= log_partition_coefficient);
            bpp[i][j] += exp(newscore + Nio_[j][i].beta - log_partition_coefficient); // TBType::N_HAIRPIN_LOOP
#else
            // C -> ( ... )
            auto newscore = st.alpha + opts.additional_paired_score(i, j);
            bpp[i][j] += exp(newscore + Cio_[j][i].beta - log_partition_coefficient); // TBType::C_HAIRPIN_LOOP
#endif
        }
        if (j==1) continue; // TODO: really need this line?

        // M: multi loop candidates
        for (const auto& [i, st]: Mio_[j])
        {
#ifdef HELIX_LENGTH
            // N -> ( M )
            auto newscore = st.alpha + param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            bpp[i][j] += exp(newscore + Nio_[j][i].beta - log_partition_coefficient); // TBType::N_MULTI_LOOP
#else
            // C -> ( M )
            auto newscore = st.alpha + param_->score_multi_loop(i, j) + opts.additional_paired_score(i, j);
            bpp[i][j] += exp(newscore + Cio_[j][i].beta - log_partition_coefficient); // TBType::C_MULTI_LOOP
#endif
        }

#ifdef HELIX_LENGTH
        // N: isolated closed loops
        for (const auto& [i, st]: Nio_[j])
        {
            // C -> ((( N ))) ; helix (< max_helix_length)
            ScoreType lp = ScoreType(0.);
            for (auto m=2; m<=opts.max_helix; m++)
            {
                if (i-(m-1)<1 || j+(m-1)>L || !opts.allow_paired(seq, i-(m-1), j+(m-1))) break;
                lp += opts.additional_paired_score(i-(m-1), j+(m-1));
                auto newscore = st.alpha + param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                auto p = exp(newscore + Cio_[j+(m-1)][i-(m-1)].beta - log_partition_coefficient); // TBType::C_HELIX
                for (auto k=2; k<=m; k++)
                    bpp[i-(k-1)][j+(k-1)] += p;
            }
        }

        // E: extended helices
        for (const auto& [i, st]: Eio_[j])
        {
            // E -> ( E ) ; extended helix longer than max_helix_length
            if (i-1>=1 && j+1<=L && opts.allow_paired(seq, i-1, j+1))
            {
                auto newscore = st.alpha + param_->score_single_loop(i-1, j+1, i, j) + opts.additional_paired_score(i-1, j+1);
                bpp[i-1][j+1] += exp(newscore + Eio_[j+1][i-1].beta - log_partition_coefficient); // TBType::E_HELIX
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
                auto newscore = st.alpha + param_->score_helix(i-(m-1), j+(m-1), m) + lp;
                auto p = exp(newscore + Cio_[j+(m-1)][i-(m-1)].beta - log_partition_coefficient); // TBType::C_HELIX_E
                for (auto k=2; k<=m; k++)
                    bpp[i-(k-1)][j+(k-1)] += p;
            }
        }
#endif

        // C: closed loops
        for (const auto& [i, st]: Cio_[j])
        {
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
                            auto newscore = st.alpha + param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            bpp[p][q] += exp(newscore + Nio_[q][p].beta - log_partition_coefficient); // TBType::N_INTERNAL_LOOP
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
                            auto newscore = st.alpha + param_->score_single_loop(p, q, i, j) + opts.additional_paired_score(p, q);
                            bpp[p][q] += exp(newscore + Cio_[q][p].beta - log_partition_coefficient); // TBType::C_INTERNAL_LOOP
                        }
                        q = next_pair[seq[p-1]][q];
                    }
                }
            }
#endif
        }
    }

    // std::cout << omp_get_wtime()-wtime << std::endl;
    std::vector<std::vector<std::pair<u_int32_t, float>>> bpp2(L+1);
    for (auto i=1; i!=bpp.size(); ++i)
    {
        for (const auto& [j, p]: bpp[i])
            if (p>=0.01) bpp2[i].emplace_back(j, std::min(p, 1.0f));
        std::sort(std::begin(bpp2[i]), std::end(bpp2[i]));
    }
    return bpp2;
}

// instantiation
#include "../param/turner.h"
#include "../param/contrafold.h"
#include "../param/positional_bl.h"
#include "../param/positional.h"
#include "../param/mix.h"
template class LinFold<TurnerNearestNeighbor>;
template class LinFold<CONTRAfoldNearestNeighbor>;
template class LinFold<PositionalNearestNeighborBL>;
template class LinFold<MixedNearestNeighborBL>;
template class LinFold<PositionalNearestNeighbor>;
template class LinFold<MixedNearestNeighbor>;
template class LinFold<MixedNearestNeighbor2>;
template class LinFold<MixedNearestNeighbor1D>;