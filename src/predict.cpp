//// test
#include <string>
#include "argparse.hpp"
#include "parameter.h"
#include "fasta.h"
#include "fold.h"

using namespace std::literals::string_literals;

int main(int argc, char* argv[])
{
    argparse::ArgumentParser ap(argv[0]);
    ap.add_argument("input_fasta")
        .help("FASTA-formatted input file");
    ap.add_argument("-p", "--param")
        .help("Thermodynamic parameter file")
        .default_value(""s);
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

    //std::cout << ap.get<int>("--max-bp") << std::endl;

    //auto param = std::make_unique<MaximizeBP<>>();
    //auto param = std::make_unique<MFE<>>();
    auto param = std::make_unique<MFETorch>();
#if 0
    if (ap.get<std::string>("--param").empty())
        param->load_default();
    else
        param->load(ap.get<std::string>("--param"));
#else
    param->load_default();
    //torch::save(*param, "model.pt");
 #endif
    Fold f(std::move(param));
    torch::NoGradGuard no_grad;

    auto fas = Fasta::load(ap.get<std::string>("input_fasta"));

    for (const auto& fa: fas) 
    {
        auto sc = f.compute_viterbi(fa.seq());
        std::cout << sc.item<float>() << std::endl;
        auto p = f.traceback_viterbi();
        std::string s(p.size()-1, '.');
        for (size_t i=1; i!=p.size(); ++i)
        {
            if (p[i] != 0)
                s[i-1] = p[i]>i ? '(' : ')';
        }
        std::cout << fa.seq() << std::endl << 
                s << std::endl;

        //std::cout << nussinov(fa.seq()) << std::endl;
    }
    return 0;
}