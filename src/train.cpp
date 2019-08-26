#include <string>
#include <random>
#include "argparse.hpp"
#include "parameter.h"
#include "fasta.h"
#include "fold.h"
#include "bpseq.h"

using namespace std::literals::string_literals;

int main(int argc, char* argv[])
{
    argparse::ArgumentParser ap(argv[0]);
    ap.add_argument("input_list")
        .help("the list of BPSEQ files");
    ap.add_argument("--max-bp")
        .help("maximum distance of base pairs")
        .action([](const auto& v) { return std::stoi(v); })
        .default_value(3);
    ap.add_argument("--l1-weight")
        .help("the weight for L1 regularization term")
        .action([](const auto& v) { return std::stof(v); })
        .default_value(0.0f);
    ap.add_argument("--l2-weight")
        .help("the weight for L2 regularization term")
        .action([](const auto& v) { return std::stof(v); })
        .default_value(0.0f);
    ap.add_argument("--learning-rate")
        .help("learning rate for SGD")
        .action([](const auto& v) { return std::stof(v); })
        .default_value(0.1f);
    ap.add_argument("--max-epochs")
        .help("the maximum number of epochs")
        .action([](const auto& v) { return std::stoi(v); } )
        .default_value(10);
        
    try {
        ap.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        ap.print_help();
        return 0;
    }

    auto seqs = BPSEQ::load_from_list(ap.get<std::string>("input_list").c_str());
    auto lr = ap.get<float>("--learning-rate");
    auto l1_weight = ap.get<float>("--l1-weight");
    auto l2_weight = ap.get<float>("--l2-weight");
    
    auto param = std::make_unique<MFETorch>();
    param->load_default();
    torch::optim::SGD optim(param->parameters(), lr);
    Fold<MFETorch, float> f(std::move(param) /*, 3, 100*/);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::vector<size_t> idx(seqs.size());
    std::iota(idx.begin(), idx.end(), 0);
    for (auto epoch = 0; epoch != ap.get<int>("--max-epochs"); ++epoch)
    {
        std::shuffle(idx.begin(), idx.end(), engine);
        for (auto i: idx)
        {
            auto seq = seqs[i].seq();
            auto stru = seqs[i].stru();
            optim.zero_grad();
            //std::cout << seq << std::endl << stru << std::endl;

            auto pred_score = f.compute_viterbi(seq, Fold<MFETorch, float>::penalty(stru, -1.0, +1.0));
            auto pred = f.traceback_viterbi(seq);
            
            auto ref_score = f.compute_viterbi(seq, Fold<MFETorch, float>::constraints(stru).max_internal_loop_length(seq.size()));
            auto ref = f.traceback_viterbi(seq);
            
            auto l1_reg = torch::zeros({}, torch::dtype(torch::kFloat));
            if (l1_weight > .0)
                for (const auto& m : optim.parameters())
                    l1_reg += torch::sum(torch::abs(m));

            auto l2_reg = torch::zeros({}, torch::dtype(torch::kFloat));
            if (l2_weight > .0)
                for (const auto& m : optim.parameters())
                    l2_reg += torch::sum(m * m);

            auto loss = pred - ref + l1_weight * l1_reg + l2_weight * l2_reg;
            loss.backward();
            optim.step();
            std::cout << loss.item<float>() << " " << pred_score << " " << ref_score << std::endl;
        }
    }
    
    return 0;
}