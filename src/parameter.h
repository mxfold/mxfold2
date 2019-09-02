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
        auto convert_sequence(const std::string& seq) -> SeqType;
        bool load(const char* filename);
        bool load(const std::string& filename) { return load(filename.c_str());};
        bool load_default();

        template <typename T> T hairpin(const SeqType& seq, size_t i, size_t j);
        template <typename T> T single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l);
        template <typename T> T multi_loop(const SeqType& seq, size_t i, size_t j);
        template <typename T> T multi_paired(const SeqType& seq, size_t i, size_t j);
        template <typename T> T multi_unpaired(const SeqType& seq, size_t i);
        template <typename T> T external_zero(const SeqType& seq) { return 0.0; }
        template <typename T> T external_paired(const SeqType& seq, size_t i, size_t j);
        template <typename T> T external_unpaired(const SeqType& seq, size_t i) { return 0.0; }

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

    public:
        PyMFE(pybind11::object obj);
        ~PyMFE() {};
        auto convert_sequence(const std::string& seq) -> SeqType;
        bool load(const char* filename);
        bool load(const std::string& filename) { return load(filename.c_str());};
        bool load_default();

        template <typename T> T hairpin(const SeqType& seq, size_t i, size_t j);
        template <typename T> T single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l);
        template <typename T> T multi_loop(const SeqType& seq, size_t i, size_t j);
        template <typename T> T multi_paired(const SeqType& seq, size_t i, size_t j);
        template <typename T> T multi_unpaired(const SeqType& seq, size_t i);
        template <typename T> T external_zero(const SeqType& seq) { return 0.0; }
        template <typename T> T external_paired(const SeqType& seq, size_t i, size_t j);
        template <typename T> T external_unpaired(const SeqType& seq, size_t i) { return 0.0; }

    public:
        ParamType<2> stack_;
        ParamType<1> hairpin_;
        ParamType<1> bulge_;
        ParamType<1> internal_;
        ParamType<3> mismatch_external_;
        ParamType<3> mismatch_hairpin_;
        ParamType<3> mismatch_internal_;
        ParamType<3> mismatch_internal_1n_;
        ParamType<3> mismatch_internal_23_;
        ParamType<3> mismatch_multi_;
        ParamType<4> int11_;
        ParamType<5> int21_;
        ParamType<6> int22_;
        ParamType<2> dangle5_;
        ParamType<2> dangle3_;
        ParamType<1> ml_base_;
        ParamType<1> ml_closing_;
        ParamType<1> ml_intern_;
        ParamType<1> ninio_;
        ParamType<1> max_ninio_;
        //std::map<SeqType, int> special_loops_;
        ParamType<1> duplex_init_;
        ParamType<1> terminalAU_;
        ParamType<1> lxc_;
};
