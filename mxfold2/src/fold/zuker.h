#pragma once

#include "fold.h"

#define HELIX_LENGTH
#define SIMPLE_SPARSIFICATION

template < typename P, typename S = typename P::ScoreType >
class Zuker : public Fold
{
    public:
        using ScoreType = S;

    private:
        enum TBType
        {
#ifdef HELIX_LENGTH
            N_HAIRPIN_LOOP,  // N -> ( ... )          ; hairpin loop
            N_INTERNAL_LOOP, // N -> ( ... C ... )    ; single loop
            N_MULTI_LOOP,    // N -> ( M M1 )         ; multi loop
            E_HELIX,         // E -> ( E )            ; extended helix longer than max_helix_length
            E_TERMINAL,      // E -> N                ; terminal of extended helix
            C_TERMINAL,      // C -> N                ; isolated base-pair 
            C_HELIX,         // C -> ((( N )))        ; helix (< max_helix_length)
            C_HELIX_E,       // C -> ((( E )))        ; helix (= max_helix_length)
#else
            C_HAIRPIN_LOOP,  // C -> ( ... )          ; hairpin loop
            C_INTERNAL_LOOP, // C -> ( ... C ... )    ; single loop 
            C_MULTI_LOOP,    // C -> ( M M1 )         ; multi loop
#endif
            M_PAIRED,        // M -> ... C            ; multi loop candidate
            M_BIFURCATION,   // M -> M C              ; add loop to multi loop candidate
            M_UNPAIRED,      // M -> M .              ; extend multi loop candidate
            M1_PAIRED,       // M1 -> C               ; right-most multi loop candidate
            M1_UNPAIRED,     // M1 -> M1 .            ; extend right-most multi loop candidate
            F_START,         // F -> empty            ; start external loop
            F_UNPAIRED,      // F -> . F              ; extend external loop
            F_BIFURCATION    // F -> C F              ; add loop to external loop
        };
        using TB = std::tuple<TBType, std::variant<u_int32_t, std::pair<u_int8_t, u_int8_t>>>;

    public:
        Zuker(std::unique_ptr<P>&& p);
        auto compute_viterbi(const std::string& seq, const Options& opt = Options()) -> ScoreType;
        auto traceback_viterbi() -> std::vector<u_int32_t>;
        auto traceback_viterbi(const std::string& seq, const Options& opt = Options()) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>;
        auto compute_inside(const std::string& seq, const Options& opt = Options()) -> ScoreType;
        void compute_outside(const std::string& seq, const Options& opt = Options());
        auto compute_basepairing_probabilities(const std::string& seq, const Options& opt = Options()) -> std::vector<std::vector<std::pair<u_int32_t, float>>>;
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
        TriMatrix<ScoreType> Ci_, Mi_, M1i_; 
        std::vector<ScoreType> Fi_;
        TriMatrix<ScoreType> Co_, Mo_, M1o_; 
        std::vector<ScoreType> Fo_;
#ifdef HELIX_LENGTH
        TriMatrix<ScoreType> Nv_, Ev_;
        TriMatrix<TB> Nt_, Et_;
        TriMatrix<ScoreType> Ni_, Ei_;
        TriMatrix<ScoreType> No_, Eo_;
#endif
};
