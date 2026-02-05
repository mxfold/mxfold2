#pragma once

#include "fold.h"
#include <unordered_map>

#define HELIX_LENGTH

template < typename P, typename S = typename P::ScoreType >
class LinFold : public Fold
{
    public:
        using ScoreType = S;

        struct Options : public Fold::Options {;
            Options() : Fold::Options(), beam_size_(100), use_extended_pairs_(false) {}

            Options& beam_size(u_int32_t s)
            {
                this->beam_size_ = s;
                return *this;
            }

            Options& use_extended_pairs(bool use)
            {
                this->use_extended_pairs_ = use;
                return *this;
            }

            auto beam_size() const { return beam_size_; }
            auto use_extended() const { return use_extended_pairs_; }
            auto make_constraint(const std::string& seq, std::string alphabests="acguACGU"s, bool canonical_only=true) const
                -> std::tuple<std::vector<std::vector<u_int32_t>>, std::vector<u_int32_t>, std::vector<bool>>;
            // Extended version using base_id hash map for Unicode support
            auto make_constraint_extended(const std::string& seq, bool canonical_only=true) const
                -> std::tuple<std::unordered_map<base_id, std::vector<u_int32_t>>, std::vector<u_int32_t>, std::vector<bool>, std::vector<base_id>>;

            u_int32_t beam_size_;
            bool use_extended_pairs_;
        };

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

        struct AlphaBeta
        {
            AlphaBeta() : alpha(std::numeric_limits<ScoreType>::lowest()), beta(std::numeric_limits<ScoreType>::lowest()) {}

            ScoreType alpha, beta;
        };

        // Structures for OpenMP parallelization
        struct PendingUpdate
        {
            u_int32_t target_j;  // Target position for update
            u_int32_t key;       // Hash map key
            ScoreType score;
            TBType manner;
            std::variant<u_int32_t, std::pair<u_int16_t, u_int16_t>> ptr;

            PendingUpdate(u_int32_t tj, u_int32_t k, ScoreType s, TBType m, u_int32_t p)
                : target_j(tj), key(k), score(s), manner(m), ptr(p) {}
            PendingUpdate(u_int32_t tj, u_int32_t k, ScoreType s, TBType m, u_int16_t p1, u_int16_t p2)
                : target_j(tj), key(k), score(s), manner(m), ptr(std::make_pair(p1, p2)) {}
        };

        struct alignas(64) ThreadLocalBuffers
        {
            std::vector<PendingUpdate> Hv_updates;
            std::vector<PendingUpdate> Nv_updates;
            std::vector<PendingUpdate> Cv_updates;
            std::vector<PendingUpdate> Mv_updates;
            std::vector<PendingUpdate> M1v_updates;
            std::vector<PendingUpdate> Ev_updates;

            void clear()
            {
                Hv_updates.clear();
                Nv_updates.clear();
                Cv_updates.clear();
                Mv_updates.clear();
                M1v_updates.clear();
                Ev_updates.clear();
            }
        };

    public:
        LinFold(std::unique_ptr<P>&& p);
        auto compute_viterbi(const std::string& seq, const Options& opt = Options()) -> ScoreType;
        auto traceback_viterbi() -> std::vector<u_int32_t>;
        auto traceback_viterbi(const std::string& seq, const Options& opt = Options()) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>;
        auto compute_inside(const std::string& seq, const Options& opt = Options()) -> ScoreType;
        void compute_outside(const std::string& seq, const Options& opt = Options());
        auto compute_basepairing_probabilities(const std::string& seq, const Options& opt = Options()) -> std::vector<std::vector<std::pair<u_int32_t, float>>>;
        const P& param_model() const { return *param_; }

    private:
        // Extended version using base_id hash map for Unicode support
        auto compute_viterbi_extended(const std::string& seq, const Options& opts) -> ScoreType;

    private:
        auto beam_prune(std::unordered_map<u_int32_t, State>& state, u_int32_t beam_size) -> ScoreType;
        auto beam_prune(std::unordered_map<u_int32_t, AlphaBeta>& state, u_int32_t beam_size) -> ScoreType;

    private:
        std::unique_ptr<P> param_;

        std::vector<std::unordered_map<u_int32_t, State>> Hv_, Cv_, Mv_, M1v_, M2v_;
#ifdef HELIX_LENGTH
        std::vector<std::unordered_map<u_int32_t, State>> Nv_, Ev_;
#endif
        std::vector<State> Fv_;

        std::vector<std::unordered_map<u_int32_t, AlphaBeta>> Hio_, Cio_, Mio_, M1io_, M2io_; 
#ifdef HELIX_LENGTH
        std::vector<std::unordered_map<u_int32_t, AlphaBeta>> Nio_, Eio_;
#endif
        std::vector<AlphaBeta> Fio_;
};
