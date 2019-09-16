#pragma once

#include <string>
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class MFE
{
    public:
        using ScoreType = float;
        using SeqType = std::vector<short>;

    private:
        using VI = std::vector<int>;
        using VVI = std::vector<VI>;
        using VVVI = std::vector<VVI>;
        using VVVVI = std::vector<VVVI>;
        using VVVVVI = std::vector<VVVVI>;
        using VVVVVVI = std::vector<VVVVVI>;

    public:
        MFE() : lxc_(107.856) {};
        ~MFE() {};
        auto convert_sequence(const std::string& seq) const -> SeqType;
        bool load(const char* filename);
        bool load(const std::string& filename) { return load(filename.c_str());};
        bool load_default();

        ScoreType score_hairpin(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l) const;
        ScoreType score_multi_loop(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_multi_paired(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_multi_unpaired(const SeqType& seq, size_t i) const;
        ScoreType score_external_zero(const SeqType& seq) const { return 0.0; }
        ScoreType score_external_paired(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_external_unpaired(const SeqType& seq, size_t i) const { return 0.0; }

        void count_hairpin(const SeqType& seq, size_t i, size_t j, ScoreType v) {}
        void count_single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l, ScoreType v) {}
        void count_multi_loop(const SeqType& seq, size_t i, size_t j, ScoreType v) {}
        void count_multi_paired(const SeqType& seq, size_t i, size_t j, ScoreType v) {}
        void count_multi_unpaired(const SeqType& seq, size_t i, ScoreType v) {}
        void count_external_zero(const SeqType& seq, ScoreType v) {}
        void count_external_paired(const SeqType& seq, size_t i, size_t j, ScoreType v) {}
        void count_external_unpaired(const SeqType& seq, size_t i, ScoreType v) {}

    public:
        VVI stack_;
        VI hairpin_;
        VI bulge_;
        VI internal_;
        VVVI mismatch_external_;
        VVVI mismatch_hairpin_;
        VVVI mismatch_internal_;
        VVVI mismatch_internal_1n_;
        VVVI mismatch_internal_23_;
        VVVI mismatch_multi_;
        VVVVI int11_;
        VVVVVI int21_;
        VVVVVVI int22_;
        VVI dangle5_;
        VVI dangle3_;
        int ml_base_;
        int ml_closing_;
        int ml_intern_;
        int ninio_;
        int max_ninio_;
        std::map<SeqType, int> special_loops_;
        int duplex_init_;
        int terminalAU_;
        float lxc_;
};

class PyMFE
{
    public:
        using ScoreType = float;
        using SeqType = std::vector<short>;

    private:
        template<int D> using ParamType = pybind11::detail::unchecked_reference<float, D> ;
        template<int D> using CountType = pybind11::detail::unchecked_mutable_reference<float, D> ;

    public:
        PyMFE(pybind11::object obj);
        ~PyMFE() {};
        auto convert_sequence(const std::string& seq) const -> SeqType;

        ScoreType score_hairpin(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l) const;
        ScoreType score_multi_loop(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_multi_paired(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_multi_unpaired(const SeqType& seq, size_t i) const;
        ScoreType score_external_zero(const SeqType& seq) const { return 0.0; }
        ScoreType score_external_paired(const SeqType& seq, size_t i, size_t j) const;
        ScoreType score_external_unpaired(const SeqType& seq, size_t i) const { return 0.0; }

        void count_hairpin(const SeqType& seq, size_t i, size_t j, ScoreType v);
        void count_single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l, ScoreType v);
        void count_multi_loop(const SeqType& seq, size_t i, size_t j, ScoreType v);
        void count_multi_paired(const SeqType& seq, size_t i, size_t j, ScoreType v);
        void count_multi_unpaired(const SeqType& seq, size_t i, ScoreType v);
        void count_external_zero(const SeqType& seq, ScoreType v) { }
        void count_external_paired(const SeqType& seq, size_t i, size_t j, ScoreType v);
        void count_external_unpaired(const SeqType& seq, size_t i, ScoreType v) { }

    private:
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
};