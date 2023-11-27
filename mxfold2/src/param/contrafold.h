#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class CONTRAfoldNearestNeighbor
{
    public:
        using ScoreType = float;

    private:
        using SeqType = std::vector<short>;
        template<int D> using ParamType = pybind11::detail::unchecked_reference<float, D> ;
        template<int D> using CountType = pybind11::detail::unchecked_mutable_reference<float, D> ;

    public:
        CONTRAfoldNearestNeighbor(const std::string& seq, pybind11::object obj);
        ~CONTRAfoldNearestNeighbor() {};

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
        static auto convert_sequence(const std::string& seq) -> SeqType;

#if 0
        ScoreType score_base_pair(short i, short j) const;
        ScoreType score_helix_stacking(short i, short j, short k, short l) const;
        ScoreType score_internal_1x1(short i, short j) const;
#endif
        void cache_count_base_pair(short i, short j, ScoreType v);
        void cache_count_helix_stacking(short i, short j, short k, short l, ScoreType v);
        void cache_count_internal_1x1(short i, short j, ScoreType v);

    private:
        SeqType seq2_;

        ParamType<2> score_base_pair_;
        ParamType<4> score_terminal_mismatch_;
        ParamType<1> score_hairpin_length_;
        ParamType<2> score_internal_explicit_;
        ParamType<1> score_bulge_length_;
        ParamType<1> score_internal_length_;
        ParamType<1> score_internal_symmetry_;
        ParamType<1> score_internal_asymmetry_;
        ParamType<1> score_bulge_0x1_;
        ParamType<2> score_internal_1x1_;
        ParamType<4> score_helix_stacking_;
        ParamType<2> score_helix_closing_;
        ParamType<1> score_multi_base_;
        ParamType<1> score_multi_unpaired_;
        ParamType<1> score_multi_paired_;
        ParamType<3> score_dangle_left_;
        ParamType<3> score_dangle_right_;
        ParamType<1> score_external_unpaired_;
        ParamType<1> score_external_paired_;

        CountType<2> count_base_pair_;
        CountType<4> count_terminal_mismatch_;
        CountType<1> count_hairpin_length_;
        CountType<2> count_internal_explicit_;
        CountType<1> count_bulge_length_;
        CountType<1> count_internal_length_;
        CountType<1> count_internal_symmetry_;
        CountType<1> count_internal_asymmetry_;
        CountType<1> count_bulge_0x1_;
        CountType<2> count_internal_1x1_;
        CountType<4> count_helix_stacking_;
        CountType<2> count_helix_closing_;
        CountType<1> count_multi_base_;
        CountType<1> count_multi_unpaired_;
        CountType<1> count_multi_paired_;
        CountType<3> count_dangle_left_;
        CountType<3> count_dangle_right_;
        CountType<1> count_external_unpaired_;
        CountType<1> count_external_paired_;

        std::vector<float> cache_score_hairpin_length_;
        std::vector<float> cache_score_bulge_length_;
        std::vector<float> cache_score_internal_length_;
        std::vector<float> cache_score_internal_symmetry_;
        std::vector<float> cache_score_internal_asymmetry_;
        std::vector<std::vector<float>> cache_score_base_pair_;
        std::vector<std::vector<std::vector<std::vector<float>>>> cache_score_helix_stacking_;
        std::vector<std::vector<float>> cache_score_internal_1x1_;

    public:
        const u_int32_t MAX_HAIRPIN_LENGTH;
        const u_int32_t MAX_BULGE_LENGTH;
        const u_int32_t MAX_INTERNAL_LENGTH;
        const u_int32_t MAX_SINGLE_LENGTH;
        const u_int32_t MAX_INTERNAL_SYMMETRIC_LENGTH;
        const u_int32_t MAX_INTERNAL_ASYMMETRY;
        const u_int32_t MAX_INTERNAL_EXPLICIT_LENGTH;
};
