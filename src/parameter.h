#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <torch/torch.h>

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

struct MFETorch : public torch::nn::Module
{
    using SeqType = std::vector<short>;
    using ScoreType = torch::Tensor;
    
    MFETorch();
    ~MFETorch() {}
    auto convert_sequence(const std::string& seq) -> SeqType;
    bool load_default();

    auto hairpin(const SeqType& seq, size_t i, size_t j) -> ScoreType;
    auto single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l) -> ScoreType;
    auto multi_loop(const SeqType& seq, size_t i, size_t j) -> ScoreType;
    auto multi_paired(const SeqType& seq, size_t i, size_t j) -> ScoreType;
    auto multi_unpaired(const SeqType& seq, size_t i) -> ScoreType;
    auto external_zero(const SeqType& seq) -> ScoreType;
    auto external_paired(const SeqType& seq, size_t i, size_t j) -> ScoreType;
    auto external_unpaired(const SeqType& seq, size_t i) -> ScoreType;

    torch::Tensor stack_;
    torch::Tensor hairpin_;
    torch::Tensor bulge_;
    torch::Tensor internal_;
    torch::Tensor mismatch_external_;
    torch::Tensor mismatch_hairpin_;
    torch::Tensor mismatch_internal_;
    torch::Tensor mismatch_internal_1n_;
    torch::Tensor mismatch_internal_23_;
    torch::Tensor mismatch_multi_;
    torch::Tensor int11_;
    torch::Tensor int21_;
    torch::Tensor int22_;
    torch::Tensor dangle5_;
    torch::Tensor dangle3_;
    torch::Tensor ml_base_;
    torch::Tensor ml_closing_;
    torch::Tensor ml_intern_;
    torch::Tensor ninio_;
    torch::Tensor max_ninio_;
    //std::map<SeqType, int> special_loops_;
    torch::Tensor duplex_init_;
    torch::Tensor terminalAU_;
    torch::Tensor lxc_;
};