#pragma once

#include "fold.h"

#define HELIX_LENGTH

template < typename P, typename S = typename P::ScoreType >
class Zuker : public Fold
{
    public:
        using ScoreType = S;

    private:
        enum TBType
        {
#ifdef HELIX_LENGTH
            N_HAIRPIN_LOOP, N_INTERNAL_LOOP, N_MULTI_LOOP,
            E_HELIX, E_TERMINAL, 
            C_TERMINAL, C_HELIX, C_HELIX_E,
#else
            C_HAIRPIN_LOOP, C_INTERNAL_LOOP, C_MULTI_LOOP,
#endif
            M_PAIRED, M_BIFURCATION, M_UNPAIRED,
            M1_PAIRED, M1_UNPAIRED,
            F_START, F_UNPAIRED, F_BIFURCATION
        };
        using TB = std::tuple<TBType, std::variant<u_int32_t, std::pair<u_int8_t, u_int8_t>>>;

    public:
        Zuker(std::unique_ptr<P>&& p);
        auto compute_viterbi(const std::string& seq, Options opt = Options()) -> ScoreType;
        auto traceback_viterbi() -> std::vector<u_int32_t>;
        auto traceback_viterbi(const std::string& seq, Options opt = Options()) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>;
        const P& param_model() const { return *param_; }

    private:
        bool update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int32_t k=0);
        bool update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int8_t p, u_int8_t q);

    private:
        std::unique_ptr<P> param_;
        TriMatrix<ScoreType> Cv_, Mv_, M1v_; 
        std::vector<ScoreType> Fv_;
        TriMatrix<TB> Ct_, Mt_, M1t_;
        std::vector<TB> Ft_;
#ifdef HELIX_LENGTH
        TriMatrix<ScoreType> Nv_, Ev_;
        TriMatrix<TB> Nt_, Et_;
#endif
};
