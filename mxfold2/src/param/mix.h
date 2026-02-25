#pragma once

#include <pybind11/pybind11.h>
#include "turner.h"
#include "contrafold.h"
#include "positional.h"
#include "positional_bl.h"
#include "positional_1d.h"

template <class T, class P>
class MixedNearestNeighborTempl
{
    public:
        using ScoreType = float;

    public:
        MixedNearestNeighborTempl(const std::string& seq, pybind11::object obj);
        MixedNearestNeighborTempl(const std::string& seq, pybind11::object obj,
                                  std::shared_ptr<BaseEncoding> encoding);
        ~MixedNearestNeighborTempl() {};

        auto score_hairpin(size_t i, size_t j) const -> ScoreType;
        auto score_single_loop(size_t i, size_t j, size_t k, size_t l) const -> ScoreType;
        auto score_helix(size_t i, size_t j, size_t m) const -> ScoreType;
        auto score_multi_loop(size_t i, size_t j) const -> ScoreType;
        auto score_multi_paired(size_t i, size_t j) const -> ScoreType;
        auto score_multi_unpaired(size_t i, size_t j) const -> ScoreType;
        auto score_external_zero() const  -> ScoreType;
        auto score_external_paired(size_t i, size_t j) const -> ScoreType;
        auto score_external_unpaired(size_t i, size_t j) const -> ScoreType;

        void count_hairpin(size_t i, size_t j, ScoreType v);
        void count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v);
        void count_helix(size_t i, size_t j, size_t m, ScoreType v);
        void count_multi_loop(size_t i, size_t j, ScoreType v);
        void count_multi_paired(size_t i, size_t j, ScoreType v);
        void count_multi_unpaired(size_t i, size_t j, ScoreType v);
        void count_external_zero(ScoreType v);
        void count_external_paired(size_t i, size_t j, ScoreType v);
        void count_external_unpaired(size_t i, size_t j, ScoreType v);

        // Score weight getters
        auto get_score_weight_turner() const -> ScoreType { return score_weight_turner_; }
        auto get_score_weight_positional() const -> ScoreType { return score_weight_positional_; }
        // Count weight getters
        auto get_count_weight_turner() const -> ScoreType { return count_weight_turner_; }
        auto get_count_weight_positional() const -> ScoreType { return count_weight_positional_; }

        // Score weight setters
        void set_score_weight_turner(ScoreType w) { score_weight_turner_ = w; }
        void set_score_weight_positional(ScoreType w) { score_weight_positional_ = w; }
        // Count weight setters
        void set_count_weight_turner(ScoreType w) { count_weight_turner_ = w; }
        void set_count_weight_positional(ScoreType w) { count_weight_positional_ = w; }

        // Convenience setters (sets both score and count weights)
        void set_weight_turner(ScoreType w) { score_weight_turner_ = w; count_weight_turner_ = w; }
        void set_weight_positional(ScoreType w) { score_weight_positional_ = w; count_weight_positional_ = w; }
        void set_weights(ScoreType w_turner, ScoreType w_positional) {
            score_weight_turner_ = w_turner;
            score_weight_positional_ = w_positional;
            count_weight_turner_ = w_turner;
            count_weight_positional_ = w_positional;
        }

    private:
        T turner_;
        P positional_;
        ScoreType score_weight_turner_;
        ScoreType score_weight_positional_;
        ScoreType count_weight_turner_;
        ScoreType count_weight_positional_;
};

using MixedNearestNeighbor = MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor>;
using CFMixedNearestNeighbor = MixedNearestNeighborTempl<CONTRAfoldNearestNeighbor, PositionalNearestNeighbor>;
using MixedNearestNeighborBL = MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighborBL>;
using MixedNearestNeighbor1D = MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor1D>;
