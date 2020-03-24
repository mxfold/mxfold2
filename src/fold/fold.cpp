#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <algorithm>
#include <cassert>
#include "fold.h"

//static
bool
Fold::
allow_paired(char x, char y)
{
    x = std::tolower(x);
    y = std::tolower(y);
    return (x=='a' && y=='u') || (x=='u' && y=='a') || 
        (x=='c' && y=='g') || (x=='g' && y=='c') ||
        (x=='g' && y=='u') || (x=='u' && y=='g');
}

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

auto
Fold::Options::
make_constraint(const std::string& seq, bool canonical_only /*=true*/) 
    -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>
{
    const auto L = seq.size();
    stru.reserve(L+1);
    while (stru.size() <= L)
        stru.push_back(Options::ANY);
    stru.resize(L+1);

    for (auto i=L; i>=1; i--)
        if (stru[i] > 0 && stru[i] <= L) // paired
            if ( (canonical_only && !Fold::allow_paired(seq[i-1], seq[stru[i]-1])) || // delete non-canonical base-pairs
                    (stru[i] - i <= min_hairpin) ) // delete very short hairpin
                stru[i] = stru[stru[i]] = Options::UNPAIRED;

    std::vector<bool> pk(L+1, false);
    for (auto i=1; i<=L; i++)
        if (stru[i] > 0 && stru[i] <= L) // paired
            for (auto k=i+1; k<stru[i]; k++)
                if (/*stru[k] > 0 &&*/ stru[k] <= L && stru[k] > stru[i]) // paired & pk
                    pk[i] = pk[stru[i]] = pk[k] = pk[stru[k]] = true;

    std::vector<u_int32_t> cnt(L+1, 0);
    for (auto p: stru2) 
    {
        const auto [p1, p2] = std::minmax(p.first, p.second);
        if (p2-p1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[p1-1], seq[p2-1])))
        {
            cnt[p1]++;
            cnt[p2]++;
        }
    }
    std::vector<bool> just_once_paired(L+1, false);
    for (auto p: stru2)
    {
        if (cnt[p.first]==1 && cnt[p.second]==1)
        {
            just_once_paired[p.first] = true;
            just_once_paired[p.second] = true;
        }
    }

    for (auto p: stru2)
    {
        const auto [p1, p2] = std::minmax(p.first, p.second);
        if (p2-p1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[p1-1], seq[p2-1])))
        {
            for (auto q: stru2)
            {
                if (p==q) continue;
                const auto [q1, q2] = std::minmax(q.first, q.second);
                if (q2-q1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[q1-1], seq[q2-1])))
                    if (p1 < q1 && q1 < p2 && p2 < q2)
                        pk[p1] = pk[p2] = pk[q1] = pk[q2] = true;
            }
        }
    }

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1, false));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1, false));
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired[i][i-1] = true; // the empty string is alway allowed to be unpaired
        allow_unpaired[i][i] = stru[i]==Options::ANY || stru[i]==Options::UNPAIRED || pk[i];
        if (just_once_paired[i] && !pk[i])
            allow_unpaired[i][i] = false;
        bool bp_l = stru[i]==Options::ANY || stru[i]==Options::PAIRED_L || stru[i]==Options::PAIRED_LR;
        for (auto j=i+1; j<=L; j++)
        {
            allow_paired[i][j] = j-i > min_hairpin;
            bool bp_r = stru[j]==Options::ANY || stru[j]==Options::PAIRED_R || stru[j]==Options::PAIRED_LR;
            allow_paired[i][j] = allow_paired[i][j] && ((bp_l && bp_r) || stru[i]==j);
            if (canonical_only)
                allow_paired[i][j] = allow_paired[i][j] && Fold::allow_paired(seq[i-1], seq[j-1]);
            allow_unpaired[i][j] = allow_unpaired[i][j-1] && allow_unpaired[j][j];
        }
    }

    for (auto p: stru2)
    {
        const auto [p1, p2] = std::minmax(p.first, p.second);
        if (p2-p1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[p1-1], seq[p2-1])))
            allow_paired[p1][p2] = true;
    }

    return std::make_pair(allow_paired, allow_unpaired);
}

auto 
Fold::Options::
make_penalty(size_t L) 
    -> std::tuple<TriMatrix<float>, std::vector<std::vector<float>>, float>
{
    TriMatrix<float> p_paired(L+1, 0.0);
    std::vector<std::vector<float>> p_unpaired(L+1, std::vector<float>(L+1, 0.0));
    float p_const = 0;
    if (use_penalty)
    {
        if (ref2.size() == 0)
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
        }            
        else
        {
            std::vector<bool> paired(L+1, false);
            for (auto p: ref2)
                paired[p.first] = paired[p.second] = true;

            for (auto i=L; i>=1; i--)
            {
                if (!paired[i])
                {
                    p_unpaired[i][i] = -pos_unpaired;
                    p_const += pos_unpaired;
                }
                else
                    p_unpaired[i][i] = neg_unpaired;
                
                for (auto j=i+1; j<=L; j++)
                {
                    p_unpaired[i][j] = p_unpaired[i][j-1] + p_unpaired[j][j];
                    p_paired[i][j] = neg_paired;
                }
            }
            for (auto p: ref2)
            {
                const auto [p1, p2] = std::minmax(p.first, p.second);
                p_paired[p1][p2] = -pos_paired;
                p_const += pos_paired;
            }
        }
    }
    return std::make_tuple(p_paired, p_unpaired, p_const);
}
