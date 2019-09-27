#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class TurnerNearestNeighbor
{
    public:
        using ScoreType = float;

    private:
        using SeqType = std::vector<short>;
        template<int D> using ParamType = pybind11::detail::unchecked_reference<float, D> ;
        template<int D> using CountType = pybind11::detail::unchecked_mutable_reference<float, D> ;

    public:
        TurnerNearestNeighbor(const std::string& seq, pybind11::object obj);
        ~TurnerNearestNeighbor() {};

        ScoreType score_hairpin(size_t i, size_t j) const;
        ScoreType score_single_loop(size_t i, size_t j, size_t k, size_t l) const;
        ScoreType score_multi_loop(size_t i, size_t j) const;
        ScoreType score_multi_paired(size_t i, size_t j) const;
        ScoreType score_multi_unpaired(size_t i) const;
        ScoreType score_external_zero() const { return 0.0; }
        ScoreType score_external_paired(size_t i, size_t j) const;
        ScoreType score_external_unpaired(size_t i) const { return 0.0; }

        void count_hairpin(size_t i, size_t j, ScoreType v);
        void count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v);
        void count_multi_loop(size_t i, size_t j, ScoreType v);
        void count_multi_paired(size_t i, size_t j, ScoreType v);
        void count_multi_unpaired(size_t i, ScoreType v);
        void count_external_zero(ScoreType v) { }
        void count_external_paired(size_t i, size_t j, ScoreType v);
        void count_external_unpaired(size_t i, ScoreType v) { }

    private:
        static auto convert_sequence(const std::string& seq) -> SeqType;

    private:
        SeqType seq2_;

        bool use_score_hairpin_at_least_;
        bool use_score_bulge_at_least_;
        bool use_score_internal_at_least_;

        ParamType<2> score_stack_;
        ParamType<1> score_hairpin_;
        ParamType<1> score_bulge_;
        ParamType<1> score_internal_;
        ParamType<3> score_mismatch_external_;
        ParamType<3> score_mismatch_hairpin_;
        ParamType<3> score_mismatch_internal_;
        ParamType<3> score_mismatch_internal_1n_;
        ParamType<3> score_mismatch_internal_23_;
        ParamType<3> score_mismatch_multi_;
        ParamType<4> score_int11_;
        ParamType<5> score_int21_;
        ParamType<6> score_int22_;
        ParamType<2> score_dangle5_;
        ParamType<2> score_dangle3_;
        ParamType<1> score_ml_base_;
        ParamType<1> score_ml_closing_;
        ParamType<1> score_ml_intern_;
        ParamType<1> score_ninio_;
        ParamType<1> score_max_ninio_;
        ParamType<1> score_duplex_init_;
        ParamType<1> score_terminalAU_;
        ParamType<1> score_lxc_;

        bool use_count_hairpin_at_least_;
        bool use_count_bulge_at_least_;
        bool use_count_internal_at_least_;
        CountType<2> count_stack_;
        CountType<1> count_hairpin_;
        CountType<1> count_bulge_;
        CountType<1> count_internal_;
        CountType<3> count_mismatch_external_;
        CountType<3> count_mismatch_hairpin_;
        CountType<3> count_mismatch_internal_;
        CountType<3> count_mismatch_internal_1n_;
        CountType<3> count_mismatch_internal_23_;
        CountType<3> count_mismatch_multi_;
        CountType<4> count_int11_;
        CountType<5> count_int21_;
        CountType<6> count_int22_;
        CountType<2> count_dangle5_;
        CountType<2> count_dangle3_;
        CountType<1> count_ml_base_;
        CountType<1> count_ml_closing_;
        CountType<1> count_ml_intern_;
        CountType<1> count_ninio_;
        CountType<1> count_max_ninio_;
        CountType<1> count_duplex_init_;
        CountType<1> count_terminalAU_;
        CountType<1> count_lxc_;

        std::vector<float> cache_score_hairpin_;
        std::vector<float> cache_score_bulge_;
        std::vector<float> cache_score_internal_;

    private:
        static int complement_pair[5][5];
};

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
        ScoreType score_multi_loop(size_t i, size_t j) const;
        ScoreType score_multi_paired(size_t i, size_t j) const;
        ScoreType score_multi_unpaired(size_t i) const;
        ScoreType score_external_zero() const { return 0.0; }
        ScoreType score_external_paired(size_t i, size_t j) const;
        ScoreType score_external_unpaired(size_t i) const;

        void count_hairpin(size_t i, size_t j, ScoreType v);
        void count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v);
        void count_multi_loop(size_t i, size_t j, ScoreType v);
        void count_multi_paired(size_t i, size_t j, ScoreType v);
        void count_multi_unpaired(size_t i, ScoreType v);
        void count_external_zero(ScoreType v) { }
        void count_external_paired(size_t i, size_t j, ScoreType v);
        void count_external_unpaired(size_t i, ScoreType v);

    private:
        ParamType<2> score_base_pair_;
        CountType<2> count_base_pair_;
        ParamType<2> score_helix_stacking_;
        CountType<2> count_helix_stacking_;
        ParamType<2> score_helix_closing_;
        CountType<2> count_helix_closing_;
        ParamType<2> score_mismatch_external_;
        CountType<2> count_mismatch_external_;
        ParamType<2> score_mismatch_hairpin_;
        CountType<2> count_mismatch_hairpin_;
        ParamType<2> score_mismatch_internal_;
        CountType<2> count_mismatch_internal_;
        ParamType<2> score_mismatch_multi_;
        CountType<2> count_mismatch_multi_;

        ParamType<1> score_base_hairpin_;
        CountType<1> count_base_hairpin_;
        ParamType<1> score_base_internal_;
        CountType<1> count_base_internal_;
        ParamType<1> score_base_multi_;
        CountType<1> count_base_multi_;
        ParamType<1> score_base_external_;
        CountType<1> count_base_external_;

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
};