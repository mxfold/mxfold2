#pragma once

#include "fold.h"
#include <unordered_map>

#define HELIX_LENGTH

template < typename P, typename S = typename P::ScoreType >
class LinFold : public Fold
{
    public:
        using ScoreType = S;

    private:
        enum TBType
        {
            H_CLOSING,       // H -> [...]            ; hairpin candidate
                             // H -> [H ...]          ; extend hairpin candidate
#ifdef HELIX_LENGTH
            N_HAIRPIN_LOOP,  // N -> [ H ]            ; hairpin loop
            N_INTERNAL_LOOP, // N -> ( ... C ... )    ; single loop
            N_MULTI_LOOP,    // N -> [ M ]            ; multi loop
            E_HELIX,         // E -> ( E )            ; extended helix longer than max_helix_length
            E_TERMINAL,      // E -> N                ; terminal of extended helix
            C_TERMINAL,      // C -> N                ; isolated base-pair 
            C_HELIX,         // C -> ((( N )))        ; helix (< max_helix_length)
            C_HELIX_E,       // C -> ((( E )))        ; helix (= max_helix_length)
#else
            C_HAIRPIN_LOOP,  // C -> [ H ]            ; hairpin loop
            C_INTERNAL_LOOP, // C -> ( ... C ... )    ; single loop 
            C_MULTI_LOOP,    // C -> [ M ]            ; multi loop
#endif
            M_CLOSING,       // M -> ... M2 ...       ; multi loop candidate
                             // M -> M ...            ; extend multi loop candidate
            M2_BIFURCATION,  // M2 -> M1 C            ; add loop to multi loop candidate
            M1_M2,           // M1 -> M2              ; multi loop candidate without unpaired 
            M1_PAIRED,       // M1 -> C               ; right-most multi loop candidate
            M1_UNPAIRED,     // M1 -> M1 .            ; extend right-most multi loop candidate
            F_START,         // F -> empty            ; start external loop
            F_UNPAIRED,      // F -> F .              ; extend external loop
            F_BIFURCATION,   // F -> F C              ; add loop to external loop
            NONE
        };

        struct State
        {
            State() : score(std::numeric_limits<ScoreType>::lowest()), manner(NONE) {}
            State(ScoreType s, TBType t, u_int32_t k=0) : score(s), manner(t), ptr(k) {}  
            State(ScoreType s, TBType t, u_int16_t p, u_int16_t q) : score(s), manner(t), ptr(std::make_pair(p, q)) {}
            bool update_max(ScoreType s, TBType t, u_int32_t k=0);
            bool update_max(ScoreType s, TBType t, u_int16_t p, u_int16_t q);

            ScoreType score;
            TBType manner;
            std::variant<u_int32_t, std::pair<u_int16_t, u_int16_t>> ptr;
        };

    public:
        LinFold(std::unique_ptr<P>&& p);
        auto compute_viterbi(const std::string& seq, const Options& opt = Options()) -> ScoreType;
        auto traceback_viterbi() -> std::vector<u_int32_t>;
        auto traceback_viterbi(const std::string& seq, const Options& opt = Options()) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>;
        auto compute_inside(const std::string& seq, const Options& opt = Options()) -> ScoreType;
        void compute_outside(const std::string& seq, const Options& opt = Options());
        auto compute_basepairing_probabilities(const std::string& seq, const Options& opt = Options()) -> std::vector<std::vector<float>>;
        const P& param_model() const { return *param_; }

    private:
        auto beam_prune(std::unordered_map<u_int32_t, State>& state, u_int32_t beam_size) -> ScoreType;

    private:
        std::unique_ptr<P> param_;

        std::vector<std::unordered_map<u_int32_t, State>> Hv_, Cv_, Mv_, M1v_, M2v_;
        std::vector<State> Fv_;
        std::vector<std::unordered_map<u_int32_t, State>> Ho_, Co_, Mo_, M1o_, M2o_; 
        std::vector<State> Fo_;
#ifdef HELIX_LENGTH
        std::vector<std::unordered_map<u_int32_t, State>> Nv_, Ev_;
        std::vector<std::unordered_map<u_int32_t, State>> No_, Eo_;
#endif

        std::vector<std::vector<u_int32_t>> next_pair_;
};
