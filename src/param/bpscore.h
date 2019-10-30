#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class PositionalBasePairScore
{
    public:
        using ScoreType = float;

    private:
        template<int D> using ParamType = pybind11::detail::unchecked_reference<float, D> ;
        template<int D> using CountType = pybind11::detail::unchecked_mutable_reference<float, D> ;

    public:
        PositionalBasePairScore(const std::string& seq, pybind11::object obj);
        ~PositionalBasePairScore() {};

        ScoreType score_paired(size_t i, size_t j) const;
        ScoreType score_unpaired(size_t i) const;

        void count_paired(size_t i, size_t j, ScoreType v);
        void count_unpaired(size_t i, ScoreType);

    private:
        ParamType<2> score_paired_;
        ParamType<1> score_unpaired_;
        CountType<2> count_paired_;
        CountType<1> count_unpaired_;
};