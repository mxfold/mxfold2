#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class PositionalNearestNeighbor
{
    public:
        using ScoreType = float;

    private:
        template<int D> using ParamType = pybind11::detail::unchecked_reference<float, D> ;
        template<int D> using CountType = pybind11::detail::unchecked_mutable_reference<float, D> ;

    public:
        PositionalNearestNeighbor(const std::string& seq, pybind11::object obj);
        ~PositionalNearestNeighbor() {};

        ScoreType score_hairpin(size_t i, size_t j) const;
        ScoreType score_single_loop(size_t i, size_t j, size_t k, size_t l) const;
        ScoreType score_helix(size_t i, size_t j, size_t m) const;
        ScoreType score_multi_loop(size_t i, size_t j) const;
        ScoreType score_multi_paired(size_t i, size_t j) const;
        ScoreType score_multi_unpaired(size_t i, size_t j) const;
        ScoreType score_external_zero() const { return 0.0; }
        ScoreType score_external_paired(size_t i, size_t j) const;
        ScoreType score_external_unpaired(size_t i, size_t j) const;

        void count_hairpin(size_t i, size_t j, ScoreType v);
        void count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v);
        void count_helix(size_t i, size_t j, size_t m, ScoreType v);
        void count_multi_loop(size_t i, size_t j, ScoreType v);
        void count_multi_paired(size_t i, size_t j, ScoreType v);
        void count_multi_unpaired(size_t i, size_t j, ScoreType v);
        void count_external_zero(ScoreType v) { }
        void count_external_paired(size_t i, size_t j, ScoreType v);
        void count_external_unpaired(size_t i, size_t j, ScoreType v);

    private:
        ParamType<2> score_basepair_;
        CountType<2> count_basepair_;
        ParamType<2> score_helix_stacking_;
        CountType<2> count_helix_stacking_;
        ParamType<2> score_mismatch_external_;
        CountType<2> count_mismatch_external_;
        ParamType<2> score_mismatch_hairpin_;
        CountType<2> count_mismatch_hairpin_;
        ParamType<2> score_mismatch_internal_;
        CountType<2> count_mismatch_internal_;
        ParamType<2> score_mismatch_multi_;
        CountType<2> count_mismatch_multi_;

        ParamType<2> score_base_hairpin_;
        CountType<2> count_base_hairpin_;
        ParamType<2> score_base_internal_;
        CountType<2> count_base_internal_;
        ParamType<2> score_base_multi_;
        CountType<2> count_base_multi_;
        ParamType<2> score_base_external_;
        CountType<2> count_base_external_;

        ParamType<1> score_hairpin_length_;
        CountType<1> count_hairpin_length_;
        ParamType<1> score_bulge_length_;
        CountType<1> count_bulge_length_;
        ParamType<1> score_internal_length_;
        CountType<1> count_internal_length_;
        ParamType<2> score_internal_explicit_;
        CountType<2> count_internal_explicit_;
        ParamType<1> score_internal_symmetry_;
        CountType<1> count_internal_symmetry_;
        ParamType<1> score_internal_asymmetry_;
        CountType<1> count_internal_asymmetry_;
        ParamType<1> score_helix_length_;
        CountType<1> count_helix_length_;

    public:
        const u_int32_t MAX_HAIRPIN_LENGTH;
        const u_int32_t MAX_BULGE_LENGTH;
        const u_int32_t MAX_INTERNAL_LENGTH;
        const u_int32_t MAX_SINGLE_LENGTH;
        const u_int32_t MAX_INTERNAL_SYMMETRIC_LENGTH;
        const u_int32_t MAX_INTERNAL_ASYMMETRY;
        const u_int32_t MAX_INTERNAL_EXPLICIT_LENGTH;
        const u_int32_t MAX_HELIX_LENGTH;
};