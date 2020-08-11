#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <cassert>
#include "nussinov.h"

template < typename P, typename S >
Nussinov<P, S>::
Nussinov(std::unique_ptr<P>&& p)
    : param_(std::move(p))
{

}

template < typename P, typename S >
auto
Nussinov<P, S>::
compute_viterbi(const std::string& seq, Options opts /*= Options()*/) -> ScoreType
{
    const auto L = seq.size();
    const ScoreType NEG_INF = std::numeric_limits<ScoreType>::lowest();
    Dv_.clear(); Dv_.resize(L+1, NEG_INF);
    Dt_.clear(); Dt_.resize(L+1);

    const auto [allow_paired, allow_unpaired] = opts.make_constraint(seq);
    const auto [loss_paired, loss_unpaired, loss_const] = opts.make_penalty(L);

    std::vector<std::vector<u_int32_t>> split_point_l(L+1);

    for (auto i=L; i>=1; i--)
    {
        if (allow_unpaired[i][i])
        {
            Dv_[i][i] = loss_unpaired[i][i] + param_->score_unpaired(i);
            Dt_[i][i] = {TB_L, 0};
        }

        for (auto j=i+1; j<=L; j++)
        {
            if (allow_unpaired[i][i] /* && i+1<j */)
            {
                auto s = Dv_[i+1][j] + loss_unpaired[i][i] + param_->score_unpaired(i);
                if (Dv_[i][j] < s)
                {
                    Dv_[i][j] = s;
                    Dt_[i][j] = {TB_L, 0};
                }
            }

            if (allow_unpaired[j][j] /* && i<j-1 */)
            {
                auto s = Dv_[i][j-1] + loss_unpaired[j][j] + param_->score_unpaired(j);
                if (Dv_[i][j] < s)
                {
                    Dv_[i][j] = s;
                    Dt_[i][j] = {TB_R, 0};
                }
            }

            //for (auto k=i+1; k<j; ++k) 
            for (auto k: split_point_l[j])
            {
                auto s = Dv_[i][k-1] + Dv_[k][j];
                if (Dv_[i][j] < s)
                {
                    Dv_[i][j] = s;
                    Dt_[i][j] = {TB_B, k};
                }
            }

            if (allow_paired[i][j] /* && i+1<j-1 */)
            {
                auto s = Dv_[i+1][j-1] + loss_paired[i][j] + param_->score_paired(i, j);
                if (Dv_[i][j] < s)
                {
                    Dv_[i][j] = s;
                    Dt_[i][j] = {TB_P, 0u};
                    split_point_l[j].push_back(i);
                }
            }

        }
    }

    return Dv_[1][L] + loss_const;
}

template < typename P, typename S >
auto
Nussinov<P, S>::
traceback_viterbi() -> std::vector<u_int32_t>
{
    const auto L = Dt_.size()-1;
    std::vector<u_int32_t> pair(L+1, 0);
    std::queue<std::tuple<TB, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Dt_[1][L], 1, L);

    while (!tb_queue.empty())
    {
        const auto [tb, i, j] = tb_queue.front();
        const auto [tb_type, k] = tb;
        tb_queue.pop();

        switch (tb_type)
        {
            case TB_P: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                if (i+1<=j-1)
                    tb_queue.emplace(Dt_[i+1][j-1], i+1, j-1);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TB_L: {
                if (i+1<=j)
                    tb_queue.emplace(Dt_[i+1][j], i+1, j);
                break;
            }
            case TB_R: {
                if (i<=j-1)
                    tb_queue.emplace(Dt_[i][j-1], i, j-1);
                break;
            }
            case TB_B: {
                tb_queue.emplace(Dt_[i][k-1], i, k-1);
                tb_queue.emplace(Dt_[k][j], k, j);
                break;
            }
        }
    }

    return pair;
}

template < typename P, typename S >
auto
Nussinov<P, S>::
traceback_viterbi(const std::string& seq, Options opts /*= Options()*/) -> std::pair<typename P::ScoreType, std::vector<u_int32_t>>
{
    const auto L = Dt_.size()-1;
    std::vector<u_int32_t> pair(L+1, 0);
    const auto [loss_paired, loss_unpaired, loss_const] = opts.make_penalty(L);
    std::queue<std::tuple<TB, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Dt_[1][L], 1, L);
    auto e = 0.;

    while (!tb_queue.empty())
    {
        const auto [tb, i, j] = tb_queue.front();
        const auto [tb_type, k] = tb;
        tb_queue.pop();

        switch (tb_type)
        {
            case TB_P: {
                assert(pair[i] == 0);
                assert(pair[j] == 0);
                if (i+1<=j-1)
                    tb_queue.emplace(Dt_[i+1][j-1], i+1, j-1);
                e += loss_paired[i][j] + param_->score_paired(i, j);
                param_->count_paired(i, j, 1);
                pair[i] = j;
                pair[j] = i;
                break;
            }
            case TB_L: {
                if (i+1<=j)
                    tb_queue.emplace(Dt_[i+1][j], i+1, j);
                e += loss_unpaired[i][i] + param_->score_unpaired(i);
                param_->count_unpaired(i, 1);
                break;
            }
            case TB_R: {
                if (i<=j-1)
                    tb_queue.emplace(Dt_[i][j-1], i, j-1);
                e += loss_unpaired[j][j] + param_->score_unpaired(j);
                param_->count_unpaired(j, 1);
                break;
            }
            case TB_B: {
                tb_queue.emplace(Dt_[i][k-1], i, k-1);
                tb_queue.emplace(Dt_[k][j], k, j);
                break;
            }
        }
    }

    return std::make_pair(e + loss_const, pair);
}

#ifdef TEST
class SimpleNussinovScore
{
    public:
        using ScoreType = float;

    public:
        SimpleNussinovScore(const std::string& seq) 
            :   score_paired_(seq.size()+1, std::vector<float>(seq.size()+1, 0.)),
                score_unpaired_(seq.size()+1, 0.)
        {
            for (auto i=seq.size(); i>=1; i--)
                for (auto j=i+1; j<seq.size()+1; j++)
                    if (Fold::allow_paired(seq[i-1], seq[j-1]))
                        score_paired_[i][j] = 1.;
                    else
                        score_paired_[i][j] = -1.;
        }

        ScoreType score_paired(u_int32_t i, u_int32_t j) const { return score_paired_[i][j]; }
        ScoreType score_unpaired(u_int32_t i) const { return score_unpaired_[i]; }


    private:
        std::vector<std::vector<float>> score_paired_;
        std::vector<float> score_unpaired_;
};

int main()
{
    std::string seq="gaaaaaaaaaaauuuuuuggggggggccccccccuuuu";
    auto p = std::make_unique<SimpleNussinovScore>(seq);
    Nussinov<SimpleNussinovScore> fold(std::move(p));
    auto s = fold.compute_viterbi(seq);
    auto x = fold.traceback_viterbi();
    auto r = Nussinov<SimpleNussinovScore>::make_paren(x);
    std::cout << seq << std::endl 
        << r << " (" << s << ")" << std::endl;
    return 0;
}
#endif

#include "../param/bpscore.h"

template class Nussinov<PositionalBasePairScore>;