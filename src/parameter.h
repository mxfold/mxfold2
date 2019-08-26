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

        static ScoreType NEG_INF() 
        {
            return std::numeric_limits<ScoreType>::lowest();
        }

        static ScoreType ZERO()
        {
            return 0.;
        }

        template <typename T> T hairpin(const SeqType& seq, size_t i, size_t j);
        template <typename T> T single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l);
        template <typename T> T multi_loop(const SeqType& seq, size_t i, size_t j);
        template <typename T> T multi_paired(const SeqType& seq, size_t i, size_t j);
        template <typename T> T multi_unpaired(const SeqType& seq, size_t i);
        template <typename T> T external_zero(const SeqType& seq) { return 0.0; }
        template <typename T> T external_paired(const SeqType& seq, size_t i, size_t j);
        template <typename T> T external_unpaired(const SeqType& seq, size_t i) { return 0.0; }

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

struct MFETorch : public torch::nn::Module
{
    using SeqType = std::vector<short>;
    using ScoreType = torch::Tensor;
    
    MFETorch();
    ~MFETorch() {}
    auto convert_sequence(const std::string& seq) -> SeqType;
    bool load_default();

    static ScoreType NEG_INF() 
    {
        return torch::full({}, std::numeric_limits<float>::lowest(), torch::requires_grad(false));
    }

    static ScoreType ZERO()
    {
        return torch::zeros({}, torch::dtype(torch::kFloat).requires_grad(false));
    }

    template <typename T> T hairpin(const SeqType& seq, size_t i, size_t j);
    template <typename T> T single_loop(const SeqType& seq, size_t i, size_t j, size_t k, size_t l);
    template <typename T> T multi_loop(const SeqType& seq, size_t i, size_t j);
    template <typename T> T multi_paired(const SeqType& seq, size_t i, size_t j);
    template <typename T> T multi_unpaired(const SeqType& seq, size_t i);
    template <typename T> T external_zero(const SeqType& seq);
    template <typename T> T external_paired(const SeqType& seq, size_t i, size_t j);
    template <typename T> T external_unpaired(const SeqType& seq, size_t i);

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

    torch::TensorAccessor<float, 2> stack_a_;
    torch::TensorAccessor<float, 1> hairpin_a_;
    torch::TensorAccessor<float, 1> bulge_a_;
    torch::TensorAccessor<float, 1> internal_a_;
    torch::TensorAccessor<float, 3> mismatch_external_a_;
    torch::TensorAccessor<float, 3> mismatch_hairpin_a_;
    torch::TensorAccessor<float, 3> mismatch_internal_a_;
    torch::TensorAccessor<float, 3> mismatch_internal_1n_a_;
    torch::TensorAccessor<float, 3> mismatch_internal_23_a_;
    torch::TensorAccessor<float, 3> mismatch_multi_a_;
    torch::TensorAccessor<float, 4> int11_a_;
    torch::TensorAccessor<float, 5> int21_a_;
    torch::TensorAccessor<float, 6> int22_a_;
    torch::TensorAccessor<float, 2> dangle5_a_;
    torch::TensorAccessor<float, 2> dangle3_a_;
    torch::TensorAccessor<float, 1> ml_base_a_;
    torch::TensorAccessor<float, 1> ml_closing_a_;
    torch::TensorAccessor<float, 1> ml_intern_a_;
    torch::TensorAccessor<float, 1> ninio_a_;
    torch::TensorAccessor<float, 1> max_ninio_a_;
    //std::map<SeqType, int> special_loops_;
    torch::TensorAccessor<float, 1> duplex_init_a_;
    torch::TensorAccessor<float, 1> terminalAU_a_;
    torch::TensorAccessor<float, 1> lxc_a_;
};