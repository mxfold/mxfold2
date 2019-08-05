#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include "fold.h"

bool allow_paired(char x, char y)
{
    x = std::tolower(x);
    y = std::tolower(y);
    return (x=='a' && y=='u') || (x=='u' && y=='a') || 
        (x=='c' && y=='g') || (x=='g' && y=='c') ||
        (x=='g' && y=='u') || (x=='u' && y=='g');
}

template <class P>
Fold<P>::Fold(std::unique_ptr<P>&& p, size_t min_hairpin_loop_length, size_t min_internal_loop_length)
    :   param(std::move(p)), 
        min_hairpin_loop_length_(min_hairpin_loop_length),
        min_internal_loop_length_(min_internal_loop_length)
{

}

template <class P>
bool Fold<P>::update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int32_t k)
{
    if (max_v < new_v) 
    {
        max_v = new_v;
        max_t = {tt, k};
        return true;
    }
    return false;
}

template <class P>
bool Fold<P>::update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int8_t p, u_int8_t q)
{
    if (max_v < new_v) 
    {
        max_v = new_v;
        max_t = {tt, std::make_pair(p, q)};
        return true;
    }
    return false;
}

template <class P>
auto Fold<P>::compute_viterbi(const std::string& seq)
{
    const auto seq2 = param->convert_sequence(seq);
    const auto L = seq.size();
    const ScoreType NEG_INF = std::numeric_limits<ScoreType>::lowest();
    Cv_.clear();  Cv_.resize(L+1, VI(L+1, NEG_INF));
    Mv_.clear();  Mv_.resize(L+1, VI(L+1, NEG_INF));
    M1v_.clear(); M1v_.resize(L+1, VI(L+1, NEG_INF));
    Fv_.clear();  Fv_.resize(L+1, NEG_INF);
    Ct_.clear();  Ct_.resize(L+1, VT(L+1));
    Mt_.clear();  Mt_.resize(L+1, VT(L+1));
    M1t_.clear(); M1t_.resize(L+1, VT(L+1));
    Ft_.clear();  Ft_.resize(L+1);

    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (j-i>min_hairpin_loop_length_ && allow_paired(seq[i-1], seq[j-1]))
            {
                update_max(Cv_[i][j], param->hairpin(seq2, i, j), Ct_[i][j], TBType::C_HAIRPIN_LOOP);

                for (auto k=i+1; (k-1)-(i+1)+1<min_internal_loop_length_ && k<j; k++)
                    for (auto l=j-1; ((k-1)-(i+1)+1)+((j-1)-(l+1)+1)<min_internal_loop_length_ && l-k>min_hairpin_loop_length_; l--)
                        if (allow_paired(seq[k-1], seq[l-1]))
                            update_max(Cv_[i][j], Cv_[k][l] + param->single_loop(seq2, i, j, k, l), Ct_[i][j], TBType::C_INTERNAL_LOOP, k, l);

                for (auto u=i+1; u+1<=j-1; u++)
                    update_max(Cv_[i][j], Mv_[i+1][u]+M1v_[u+1][j-1] + param->multi_loop(seq2, i, j), Ct_[i][j], TBType::C_MULTI_LOOP, u);

            }

            /////////////////
            if (j-i>min_hairpin_loop_length_ && allow_paired(seq[i-1], seq[j-1]))
                update_max(Mv_[i][j], Cv_[i][j] + param->multi_paired(seq2, i, j), Mt_[i][j], TBType::M_PAIRED, i);

            ScoreType t{0};
            for (auto u=i; u+1<j; u++)
            {
                t += param->multi_unpaired(seq2, u);
                if (j-(u+1)>min_hairpin_loop_length_ && allow_paired(seq[(u+1)-1], seq[j-1]))
                {
                    auto s = param->multi_paired(seq2, u+1, j) + t;
                    update_max(Mv_[i][j], Cv_[u+1][j]+s, Mt_[i][j], TBType::M_PAIRED, u+1);
                }
            }

            for (auto u=i; u+1<=j; u++)
                if (j-(u+1)>min_hairpin_loop_length_ && allow_paired(seq[(u+1)-1], seq[j-1]))
                    update_max(Mv_[i][j], Mv_[i][u]+Cv_[u+1][j] + param->multi_paired(seq2, u+1, j), Mt_[i][j], TBType::M_BIFURCATION, u);

            update_max(Mv_[i][j], Mv_[i][j-1] + param->multi_unpaired(seq2, j), Mt_[i][j], TBType::M_UNPAIRED);

            /////////////////
            if (j-i>min_hairpin_loop_length_ && allow_paired(seq[i-1], seq[j-1]))
                update_max(M1v_[i][j], Cv_[i][j] + param->multi_paired(seq2, i, j), M1t_[i][j], TBType::M1_PAIRED);

            update_max(M1v_[i][j], M1v_[i][j-1] + param->multi_unpaired(seq2, j), M1t_[i][j], TBType::M1_UNPAIRED);
        }
    }

    update_max(Fv_[L], param->external_zero(seq2), Ft_[L], TBType::F_ZERO);

    for (auto i=L-1; i>=1; i--)
    {
        update_max(Fv_[i], Fv_[i+1] + param->external_unpaired(seq2, i), Ft_[i], TBType::F_UNPAIRED);

        for (auto k=i+1; k+1<=L; k++)
            if (k-i>min_hairpin_loop_length_ && allow_paired(seq[i-1], seq[k-1]))
                update_max(Fv_[i], Cv_[i][k]+Fv_[k+1] + param->external_paired(seq2, i, k), Ft_[i], TBType::F_BIFURCATION, k);
    }

    update_max(Fv_[1], Cv_[1][L] + param->external_paired(seq2, 1, L), Ft_[1], TBType::F_PAIRED);

    return Fv_[1];
}

template <class P>
auto Fold<P>::traceback_viterbi()
{
    const auto L = Ft_.size()-1;
    std::vector<u_int32_t> pair(L+1, 0);
    std::queue<std::tuple<TB, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Ft_[1], 1, L);

    while (!tb_queue.empty())
    {
        const auto [tb, i, j] = tb_queue.front();
        const auto [tb_type, kl] = tb;
        tb_queue.pop();

        switch (tb_type)
        {
            case TBType::C_HAIRPIN_LOOP: {
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(kl);
                pair[p] = q;
                pair[q] = p;
                tb_queue.emplace(Ct_[p][q], p, q);
                break;
            }
            case TBType::C_MULTI_LOOP: {
                const auto k = std::get<0>(kl);
                tb_queue.emplace(Mt_[i+1][k], i+1, k);
                tb_queue.emplace(M1t_[k+1][j-1], k+1, j-1);
                break;
            }
            case TBType::M_PAIRED: {
                const auto k = std::get<0>(kl);
                pair[k] = j;
                pair[j] = k;
                tb_queue.emplace(Ct_[k][j], k, j);
                break;
            }
            case TBType::M_BIFURCATION: {
                const auto k = std::get<0>(kl);
                pair[k+1] = j;
                pair[j] = k+1;
                tb_queue.emplace(Mt_[i][k], i, k);
                tb_queue.emplace(Ct_[k+1][j], k+1, j);
                break;
            }
            case TBType::M_UNPAIRED: {
                tb_queue.emplace(Mt_[i][j-1], i, j-1);
                break;
            }    
            case TBType::M1_PAIRED: {
                pair[i] = j;
                pair[j] = i;
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                tb_queue.emplace(M1t_[i][j-1], i, j-1);
                break;
            }
            case TBType::F_ZERO: {
                break;
            }
            case TBType::F_UNPAIRED: {
                tb_queue.emplace(Ft_[i+1], i+1, j);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(kl);
                pair[i] = k;
                pair[k] = i;
                tb_queue.emplace(Ct_[i][k], i, k);
                tb_queue.emplace(Ft_[k+1], k+1, j);
                break;
            }
            case TBType::F_PAIRED: {
                pair[i] = j;
                pair[j] = i;
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
        }
    }

    return pair;
}

auto nussinov(const std::string& seq)
{
    const auto L = seq.size();
    std::vector dp(L+1, std::vector(L+1, 0));
    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (j-i>3 && allow_paired(seq[i-1], seq[j-1]))
                dp[i][j] = std::max(dp[i][j], dp[i+1][j-1]+1);
            dp[i][j] = std::max(dp[i][j], dp[i+1][j]);
            dp[i][j] = std::max(dp[i][j], dp[i][j-1]);  
            for (auto k=i+1; k<j-1; k++)
                dp[i][j]  = std::max(dp[i][j], dp[i][k]+dp[k+1][j]);
        }
    }

    return dp[1][L];
}

//// test
#include <string>
#include "argparse.hpp"
#include "parameter.h"
#include "fasta.h"

using namespace std::literals::string_literals;

int main(int argc, char* argv[])
{
    
    argparse::ArgumentParser ap(argv[0]);
    ap.add_argument("input_fasta")
        .help("FASTA-formatted input file");
    ap.add_argument("-p", "--param")
        .help("Thermodynamic parameter file")
        .default_value("rna_turner2004.par"s);
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
    auto param = std::make_unique<MFE<>>();
    param->load(ap.get<std::string>("--param"));
    Fold f(std::move(param));

    auto fas = Fasta::load(ap.get<std::string>("input_fasta"));

    for (const auto& fa: fas) 
    {
        std::cout << f.compute_viterbi(fa.seq()) << std::endl;
        auto p = f.traceback_viterbi();
        std::string s(p.size()-1, '.');
        for (size_t i=1; i!=p.size(); ++i)
        {
            if (p[i] != 0)
                s[i-1] = p[i]>i ? '(' : ')';
        }
        std::cout << fa.seq() << std::endl << 
                s << std::endl;

        std::cout << nussinov(fa.seq()) << std::endl;
    }
    return 0;
}