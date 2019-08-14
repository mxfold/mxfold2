#include <string>
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
        
    try {
        ap.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        ap.print_help();
        return 0;
    }

    auto seqs = BPSEQ::load_from_list(ap.get<std::string>("input_list").c_str());
    
    auto param = std::make_unique<MFETorch>();
    param->load_default();
    torch::optim::SGD optim(param->parameters(), .1);
    Fold f(std::move(param));

    for (const auto& s: seqs) 
    {
        optim.zero_grad();
        auto sc = f.compute_viterbi(s.seq());
        sc.backward();
        auto p = f.traceback_viterbi();
        optim.step();
#if 0
        std::cout << sc.item<float>() << std::endl;
        std::string stru(p.size()-1, '.');
        for (size_t i=1; i!=p.size(); ++i)
        {
            if (p[i] != 0)
                stru[i-1] = p[i]>i ? '(' : ')';
        }
        std::cout << s.seq() << std::endl << 
                stru << std::endl;
#endif
    }
    return 0;
}