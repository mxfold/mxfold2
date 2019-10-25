#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include <memory>
#include "trimatrix.h"

struct FoldOptions {
    size_t min_hairpin;
    size_t max_internal;
    std::string stru;
    bool use_penalty;
    std::string ref;
    float pos_paired;
    float neg_paired;
    float pos_unpaired;
    float neg_unpaired;

    FoldOptions() : 
        min_hairpin(3),
        max_internal(30),
        use_penalty(false)
    {    
    }

    FoldOptions& min_hairpin_loop_length(size_t s)
    {
        this->min_hairpin = s;
        return *this;
    }

    FoldOptions& max_internal_loop_length(size_t s)
    {
        this->max_internal = s;
        return *this;
    }

    FoldOptions& constraints(const std::string& s)
    {
        this->stru = s;
        return *this;
    }

    FoldOptions& penalty(const std::string& ref, float pos_paired=0, float neg_paired=0, float pos_unpaired=0, float neg_unpaired=0)
    {
        this->use_penalty = pos_paired!=0 || neg_paired!=0 || pos_unpaired!=0 || neg_unpaired!=0;
        this->ref = ref;
        this->pos_paired = pos_paired;
        this->neg_paired = neg_paired;
        this->pos_unpaired = pos_unpaired;
        this->neg_unpaired = neg_unpaired;
        return *this;
    }
#if 0
    static FoldOptions min_hairpin_loop_length(size_t s)
    {
        return FoldOptions().min_hairpin_loop_length(s);
    }

    static FoldOptions max_internal_loop_length(size_t s)
    {
        return FoldOtions().max_internal_loop_length(s);
    }

    static FoldOptions constraints(const std::string& s)
    {
        return FoldOptions().constraints(s);
    }

    static FoldOptions penalty(const std::string& ref, float pos_penalty, float neg_penalty)
    {
        return FoldOptions().penalty(ref, pos_penalty, neg_penalty);
    }
#endif
};



template < typename P, typename S = typename P::ScoreType >
class Fold
{
    public:
        using ScoreType = S;


    private:
        enum TBType
        {
            C_HAIRPIN_LOOP, C_INTERNAL_LOOP, C_MULTI_LOOP,
            M_PAIRED, M_BIFURCATION, M_UNPAIRED,
            M1_PAIRED, M1_UNPAIRED,
            F_START, F_UNPAIRED, F_BIFURCATION
        };
        using TB = std::tuple<TBType, std::variant<u_int32_t, std::pair<u_int8_t, u_int8_t>>>;

    public:
        Fold(std::unique_ptr<P>&& p);
        auto compute_viterbi(const std::string& seq, FoldOptions opt = FoldOptions()) -> ScoreType;
        auto traceback_viterbi() -> std::vector<u_int32_t>;
        auto traceback_viterbi(const std::string& seq, FoldOptions opt = FoldOptions()) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>;
        const P& param_model() const { return *param_; }

    private:
        bool update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int32_t k=0);
        bool update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int8_t p, u_int8_t q);

    private:
        std::unique_ptr<P> param_;
        TriMatrix<ScoreType> Cv_, Mv_, M1v_; 
        std::vector<ScoreType> Fv_;
        TriMatrix<Fold::TB> Ct_, Mt_, M1t_;
        std::vector<Fold::TB> Ft_;
};

