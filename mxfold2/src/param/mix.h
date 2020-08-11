#pragma once

#include <pybind11/pybind11.h>
#include "turner.h"
#include "positional.h"

class MixedNearestNeighbor
{
    public:
        using ScoreType = float;

    public:
        MixedNearestNeighbor(const std::string& seq, pybind11::object obj);
        ~MixedNearestNeighbor() {};

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

    private:
        TurnerNearestNeighbor turner_;
        PositionalNearestNeighbor positional_;
};
