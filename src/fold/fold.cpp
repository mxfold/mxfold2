#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
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
parse_paren(const std::string& paren) -> std::vector<u_int32_t>
{
    std::vector<u_int32_t> bp(paren.size()+1, 0);
    std::stack<u_int32_t> st;
    for (auto i=0u; i!=paren.size(); ++i)
    {
        switch (paren[i])
        {
        case '(':
            st.push(i); break;
        case ')':
        {
            auto j=st.top();
            st.pop();
            bp[i+1] = j+1;
            bp[j+1] = i+1;
        }
        break;
        default: break;
        }
    }
    return bp;
}

//static
auto
Fold::
make_constraint(const std::string& seq, std::string stru, u_int32_t max_bp, bool canonical_only /*=true*/) 
    -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>
{
    const auto L = seq.size();
    if (stru.size() < L)
        stru.append(std::string(L-stru.size(), '.'));
    else
        stru = stru.substr(0, L);
    auto bp = parse_paren(stru);

    for (auto i=L; i>=1; i--)
        if (bp[i] > 0) 
        {
            if ( (canonical_only && !allow_paired(seq[i-1], seq[bp[i]-1])) || // delete non-canonical base-pairs
                    (bp[i] - i <= max_bp) ) // delete very short hairpin
            {
                stru[i-1] = stru[bp[i]-1] = 'x';
                bp[i] = bp[bp[i]] = 0;
            }
        }

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1, false));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1, false));
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired[i][i-1] = true; // the empty string is alway allowed to be unpaired
        allow_unpaired[i][i] = stru[i-1]=='.' || stru[i-1]=='x';
        bool bp_l = stru[i-1]=='.' || stru[i-1]=='<' || stru[i-1]=='|';
        for (auto j=i+1; j<=L; j++)
        {
            allow_paired[i][j] = j-i > max_bp;
            bool bp_r = stru[j-1]=='.' || stru[j-1]=='>' || stru[j-1]=='|';
            allow_paired[i][j] = allow_paired[i][j] && ((bp_l && bp_r) || bp[i]==j);
            if (canonical_only)
                allow_paired[i][j] = allow_paired[i][j] && Fold::allow_paired(seq[i-1], seq[j-1]);
            allow_unpaired[i][j] = allow_unpaired[i][j-1] && allow_unpaired[j][j];
        }
    }
    return std::make_pair(allow_paired, allow_unpaired);
}

//static
auto 
Fold::
make_penalty(size_t L, bool use_penalty, const std::string& ref, float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired) 
    -> std::tuple<TriMatrix<float>, std::vector<std::vector<float>>, float>
{
    TriMatrix<float> p_paired(L+1, 0.0);
    std::vector<std::vector<float>> p_unpaired(L+1, std::vector<float>(L+1, 0.0));
    float p_const = 0;
    if (use_penalty)
    {
        auto bp = parse_paren(ref);
        for (auto i=L; i>=1; i--)
        {
            if (ref[i-1]=='.' || ref[i-1]=='x')
            {
                p_unpaired[i][i] = -pos_unpaired;
                p_const += pos_unpaired;
            }
            else
                p_unpaired[i][i] = neg_unpaired;

            for (auto j=i+1; j<=L; j++)
            {
                p_unpaired[i][j] = p_unpaired[i][j-1] + p_unpaired[j][j];

                if (bp[i] == j)
                {
                    p_paired[i][j] = -pos_paired;
                    p_const += pos_paired;
                }
                else
                    p_paired[i][j] = neg_paired;
            }
        }
    }
    return std::make_tuple(p_paired, p_unpaired, p_const);
}
