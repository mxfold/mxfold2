#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <map>

template <typename S = float>
class MaximizeBP
{
    public:
        using ScoreType = S;
        using SeqType = std::string;

    public:
        MaximizeBP() {};

        auto convert_sequence(const std::string& seq) -> SeqType { return seq; };
        ScoreType hairpin(const SeqType& seq, size_t i, size_t j) { return 1; }
        ScoreType single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l) { return 1; }
        ScoreType multi_loop(const SeqType& seq, size_t i, size_t j) { return 1; }
        ScoreType multi_paired(const SeqType& seq, size_t i, size_t j) { return 0; }
        ScoreType multi_unpaired(const SeqType& seq, size_t i) { return 0; }
        ScoreType external_zero(const SeqType& seq) { return 0; }
        ScoreType external_paired(const SeqType& seq, size_t i, size_t j) { return 0; }
        ScoreType external_unpaired(const SeqType& seq, size_t i) { return 0; }
};

#include <cmath>
namespace VIENNA
{
    extern "C"
    {
#include <ViennaRNA/loop_energies.h>
    }
};

template <typename S = float>
class MFEfromVienna
{
    public:
        using ScoreType = S;
        using SeqType = std::vector<short>;

    public:
        MFEfromVienna();
        ~MFEfromVienna();
        auto convert_sequence(const std::string& seq) -> SeqType;
        auto hairpin(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l) -> ScoreType;
        auto multi_loop(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto multi_paired(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto multi_unpaired(const SeqType& seq, size_t i) -> ScoreType;
        auto external_zero(const SeqType& seq) -> ScoreType { return 0.; };
        auto external_paired(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto external_unpaired(const SeqType& seq, size_t i) -> ScoreType { return 0.; };

    private:
        VIENNA::vrna_md_t md_;
        VIENNA::vrna_fold_compound_t *vc_;
};


template <typename S = float>
class MFE : public MFEfromVienna<S>
{
    public:
        using ScoreType = S;
        using SeqType = typename MFEfromVienna<S>::SeqType;

    private:
        using VI = std::vector<int>;
        using VVI = std::vector<VI>;
        using VVVI = std::vector<VVI>;
        using VVVVI = std::vector<VVVI>;
        using VVVVVI = std::vector<VVVVI>;
        using VVVVVVI = std::vector<VVVVVI>;

    public:
        MFE() : MFEfromVienna<S>(), lxc_(107.856) {};
        ~MFE() {};
        auto convert_sequence(const std::string& seq) -> SeqType;
        bool load(const char* filename);
        bool load(const std::string& filename) { return load(filename.c_str());};

        auto hairpin(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l) -> ScoreType;
        auto multi_loop(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto multi_paired(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto multi_unpaired(const SeqType& seq, size_t i) -> ScoreType;
        auto external_zero(const SeqType& seq) -> ScoreType { return 0.; };
        auto external_paired(const SeqType& seq, size_t i, size_t j) -> ScoreType;
        auto external_unpaired(const SeqType& seq, size_t i) -> ScoreType { return 0.; };

    private:
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
