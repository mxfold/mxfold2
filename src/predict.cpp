#include <string>
#include <chrono>
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
    ap.add_argument("--model")
        .help("model parameter file")
        .default_value(""s);
    ap.add_argument("--max-bp")
        .help("maximum distance of base pairs")
        .action([](const auto& v) { return std::stoi(v); })
        .default_value(3);
    ap.add_argument("--constraint")
        .help("constraint folding")
        .default_value(false)
        .implicit_value(true);

    try {
        ap.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        ap.print_help();
        return 0;
    }

    torch::NoGradGuard no_grad;
    auto param = std::make_unique<MFETorch>();
    param->eval();

    if (ap.get("--model").empty())
        param->load_default();
    else
        param->load_state_dict(ap.get("--model").c_str());

    Fold<MFETorch, float> f(std::move(param));

    auto fas = Fasta::load(ap.get<std::string>("input_fasta"));
    auto use_constraint = ap.get<bool>("--constraint");

    for (const auto& fa: fas) 
    {
        //auto start = std::chrono::system_clock::now();
        auto opts = FoldOptions();
        if (use_constraint)
            opts.constraints(fa.str()).max_internal_loop_length(fa.seq().size());
        auto sc = f.compute_viterbi(fa.seq(), opts);
        auto p = f.traceback_viterbi();
        //auto end = std::chrono::system_clock::now();
        //std::chrono::duration<double> dur = end-start;
        //std::cout << dur.count() << std::endl;

        std::string s(p.size()-1, '.');
        for (size_t i=1; i!=p.size(); ++i)
        {
            if (p[i] != 0)
                s[i-1] = p[i]>i ? '(' : ')';
            //std::cout << i << " " << fa.seq()[i-1] << " " << p[i] << std::endl;
        }

        std::cout << ">" << fa.name() << std::endl
            <<  fa.seq() << std::endl 
            << s << " (" << sc << ")" <<std::endl;
    }
    return 0;
}