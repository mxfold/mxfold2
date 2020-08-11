#pragma once

#include "fold.h"

template < typename P, typename S = typename P::ScoreType >
class Nussinov : public Fold
{
    public:
        using ScoreType = S;
    
    private:
        enum TBType { TB_P, TB_L, TB_R, TB_B };
        using TB = std::pair<TBType, u_int32_t>;

    public:
        Nussinov(std::unique_ptr<P>&& p);
        auto compute_viterbi(const std::string& seq, Options opts = Options()) -> ScoreType;
        auto traceback_viterbi() -> std::vector<u_int32_t>;
        auto traceback_viterbi(const std::string& seq, Options opt = Options()) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>;

    private:
        std::unique_ptr<P> param_;
        TriMatrix<ScoreType> Dv_;
        TriMatrix<TB> Dt_;
};
