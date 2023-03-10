#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <algorithm>
#include <cassert>
#include "fold.h"

//static
auto
Fold::
make_paren(const std::vector<u_int32_t>& p) -> std::string
{
    std::string s(p.size()-1, '.');
    for (size_t i=1; i!=p.size(); ++i)
    {
        if (p[i] != 0)
            s[i-1] = p[i]>i ? '(' : ')';
    }
    return s;
}

bool
Fold::Options::
allow_paired(char x, char y) const
{
    x = std::tolower(x);
    y = std::tolower(y);
    return allowed_pairs_[x][y];
}

auto
Fold::Options::
make_constraint(const std::string& seq, bool canonical_only /*=true*/) const
    -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>
{
    const auto L = seq.size();
    std::vector<u_int32_t> stru(L+1, Options::ANY);
    std::copy(std::begin(this->stru), std::end(this->stru), std::begin(stru));

    for (auto i=L; i>=1; i--)
        if (stru[i] > 0 && stru[i] <= L) // paired
            if ( (canonical_only && !this->allow_paired(seq[i-1], seq[stru[i]-1])) || // delete non-canonical base-pairs
                    (stru[i] - i <= min_hairpin) ) // delete very short hairpin
                stru[i] = stru[stru[i]] = Options::UNPAIRED;

    std::vector<bool> pk(L+1, false);
    for (auto i=1; i<=L; i++)
        if (stru[i] > 0 && stru[i] <= L) // paired
            for (auto k=i+1; k<stru[i]; k++)
                if (/*stru[k] > 0 &&*/ stru[k] <= L && stru[k] > stru[i]) // paired & pk
                    pk[i] = pk[stru[i]] = pk[k] = pk[stru[k]] = true;

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1, false));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1, false));
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired[i][i-1] = true; // the empty string is alway allowed to be unpaired
        allow_unpaired[i][i] = stru[i]==Options::ANY || stru[i]==Options::UNPAIRED || pk[i];
        bool bp_l = stru[i]==Options::ANY || stru[i]==Options::PAIRED_L || stru[i]==Options::PAIRED_LR;
        for (auto j=i+1; j<=L; j++)
        {
            allow_paired[i][j] = j-i > min_hairpin;
            bool bp_r = stru[j]==Options::ANY || stru[j]==Options::PAIRED_R || stru[j]==Options::PAIRED_LR;
            allow_paired[i][j] = allow_paired[i][j] && ((bp_l && bp_r) || stru[i]==j);
            if (canonical_only)
                allow_paired[i][j] = allow_paired[i][j] && this->allow_paired(seq[i-1], seq[j-1]);
            allow_unpaired[i][j] = allow_unpaired[i][j-1] && allow_unpaired[j][j];
        }
    }

    return std::make_pair(allow_paired, allow_unpaired);
}

auto 
Fold::Options::
make_additional_scores(size_t L) const
    -> std::tuple<TriMatrix<float>, std::vector<std::vector<float>>>
{
    TriMatrix<float> p_paired(L+1, 0.0);
    std::vector<std::vector<float>> p_unpaired(L+1, std::vector<float>(L+1, 0.0));

    // margin terms
    float p_const = 0;
    if (use_margin)
    {
        for (auto i=L; i>=1; i--)
        {
            if (ref[i]==Options::ANY || ref[i]==Options::UNPAIRED)
            {
                p_unpaired[i][i] = -pos_unpaired;
                p_const += pos_unpaired;
            }
            else
                p_unpaired[i][i] = neg_unpaired;

            for (auto j=i+1; j<=L; j++)
            {
                p_unpaired[i][j] = p_unpaired[i][j-1] + p_unpaired[j][j];

                if (ref[i] == j)
                {
                    p_paired[i][j] = -pos_paired;
                    p_const += pos_paired;
                }
                else
                    p_paired[i][j] = neg_paired;
            }
        }

        // pseudo-energy terms
        if (score_paired_position_.size() >= L)
            for (auto i=score_paired_position_.size(); i>=1; i--)
                for (auto j=i+1; j<=L; j++)
                    p_paired[i][j] += score_paired_position_[i-1] + score_paired_position_[j-1];
    }
    return std::make_tuple(p_paired, p_unpaired);
}