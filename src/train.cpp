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
    ap.add_argument("output_model")
        .help("the output filename of the trained model");
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
        .default_value(0.001f);
    ap.add_argument("--max-epochs")
        .help("the maximum number of epochs")
        .action([](const auto& v) { return std::stoi(v); } )
        .default_value(10);
    ap.add_argument("--random-seed")
        .help("the random seed for reproducibility")
        .action([](const auto& v) { return std::stoi(v); })
        .default_value(-1);
        
    try {
        ap.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        ap.print_help();
        return 0;
    }

    auto seed = ap.get<int>("--random-seed");
    if (seed < 0) 
    {
        std::random_device seed_gen;
        seed = seed_gen();
    }
    torch::manual_seed(seed);
    std::mt19937 engine(seed);

    auto seqs = BPSEQ::load_from_list(ap.get<std::string>("input_list").c_str());
    auto lr = ap.get<float>("--learning-rate");
    auto l1_weight = ap.get<float>("--l1-weight");
    auto l2_weight = ap.get<float>("--l2-weight");
    
    auto param = std::make_unique<MFETorch>();
    //param->load_default();
    //torch::optim::SGD optim(param->parameters(), lr);
    torch::optim::Adam optim(param->parameters(), torch::optim::AdamOptions(lr));
    Fold<MFETorch, float> f(std::move(param));

    std::vector<size_t> idx(seqs.size());
    std::iota(idx.begin(), idx.end(), 0);
    for (auto epoch = 0; epoch != ap.get<int>("--max-epochs"); ++epoch)
    {
        float loss_f = 0.;
        std::shuffle(idx.begin(), idx.end(), engine);
        for (auto i: idx)
        {
            auto seq = seqs[i].seq();
            auto stru = seqs[i].stru('x');
            optim.zero_grad();

            auto pred_opts = FoldOptions().penalty(stru, -1.0, +1.0);
            auto pred_score = f.compute_viterbi(seq, pred_opts);
            auto pred = f.traceback_viterbi(seq, pred_opts);

            auto ref_opts = FoldOptions().constraints(stru).max_internal_loop_length(seq.size());
            auto ref_score = f.compute_viterbi(seq, ref_opts);
            auto ref = f.traceback_viterbi(seq);
            
            auto loss = pred - ref;

            if (l1_weight > 0.)
            {
                auto l1_reg = torch::zeros({}, torch::dtype(torch::kFloat));
                for (const auto& m : optim.parameters())
                    l1_reg += torch::sum(torch::abs(m));
                loss += l1_weight * l1_reg;
            }

            if (l2_weight > 0.)
            {
                auto l2_reg = torch::zeros({}, torch::dtype(torch::kFloat));
                for (const auto& m : optim.parameters())
                    l2_reg += torch::sum(m * m);
                loss += l2_weight * torch::sqrt(l2_reg);
            }

            loss_f += loss.item<float>();
            loss.backward();
            optim.step();
            //std::cout << loss.item<float>() << " " << pred.item<float>() << " " << pred_score << " " << ref.item<float>() << " " << ref_score << std::endl;
        }
        std::cout << "Epoch " << epoch+1 << ": loss=" << loss_f/seqs.size() << std::endl;
    }
    f.param_model().save_state_dict(ap.get("output_model").c_str());

    return 0;
}