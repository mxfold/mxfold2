#pragma once

#include <pybind11/pybind11.h>
#include "turner.h"
#include "contrafold.h"
#include "positional.h"
#include "positional_bl.h"
#include "positional_1d.h"

template <class T, class P, int M = 1>
class MixedNearestNeighborTempl
{
    public:
        using ScoreType = float;

    public:
        MixedNearestNeighborTempl(const std::string& seq, pybind11::object obj);
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

    private:
        T turner_;
        P positional_;
};

using MixedNearestNeighbor = MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor, 1>;
using MixedNearestNeighbor2 = MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor, 2>;
using CFMixedNearestNeighbor = MixedNearestNeighborTempl<CONTRAfoldNearestNeighbor, PositionalNearestNeighbor, 1>;
using CFMixedNearestNeighbor2 = MixedNearestNeighborTempl<CONTRAfoldNearestNeighbor, PositionalNearestNeighbor, 2>;
using MixedNearestNeighborBL = MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighborBL, 1>;
using MixedNearestNeighbor1D = MixedNearestNeighborTempl<TurnerNearestNeighbor, PositionalNearestNeighbor1D, 1>;