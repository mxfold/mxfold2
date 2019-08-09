#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include <memory>
#include <torch/torch.h>
#include "parameter.h"

class Fold
{
    public:
        using ScoreType = torch::Tensor;

    private:
        enum TBType
        {
            C_HAIRPIN_LOOP, C_INTERNAL_LOOP, C_MULTI_LOOP,
            M_PAIRED, M_BIFURCATION, M_UNPAIRED,
            M1_PAIRED, M1_UNPAIRED,
            F_ZERO, F_UNPAIRED, F_BIFURCATION, F_PAIRED
        };
        using TB = std::tuple<TBType, std::variant<u_int32_t, std::pair<u_int8_t, u_int8_t>>>;

        using VI = std::vector<ScoreType>;
        using VVI = std::vector<VI>;
        using VT = std::vector<Fold::TB>;
        using VVT = std::vector<VT>;

    public:
        Fold(std::unique_ptr<MFETorch>&& p, size_t min_hairpin_loop_length=3, size_t min_internal_loop_length=30);
        auto compute_viterbi(const std::string& seq);
        auto traceback_viterbi();

    private:
        bool update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int32_t k=0);
        bool update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int8_t p, u_int8_t q);

    private:
        std::unique_ptr<MFETorch> param;
        VVI Cv_, Mv_, M1v_; 
        VI Fv_;
        VVT Ct_, Mt_, M1t_;
        VT Ft_;
        size_t min_hairpin_loop_length_;
        size_t min_internal_loop_length_;
};

