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

bool
Fold::Options::
allow_paired(const std::string& seq, u_int32_t i, u_int32_t j) const
{
    std::tie(i, j) = std::minmax(i, j);
    return j-i>min_hairpin
        && this->allow_paired(seq[i-1], seq[j-1])
        && (stru[i]==Options::ANY || stru[i]==Options::PAIRED_L || stru[i]==Options::PAIRED_LR || stru[i]==j)
        && (stru[j]==Options::ANY || stru[j]==Options::PAIRED_R || stru[j]==Options::PAIRED_LR || stru[j]==i);
}

bool
Fold::Options::
allow_paired_extended(base_id x, base_id y) const
{
    // Convert to lowercase for standard bases
    x = BaseEncoding::to_lower(x);
    y = BaseEncoding::to_lower(y);

    // 1. First check the extended pairs hash table
    uint32_t key = (static_cast<uint32_t>(x) << 16) | y;
    auto it = extended_pairs_.find(key);
    if (it != extended_pairs_.end()) {
        return it->second;
    }

    // 2. For ASCII bases, fall back to the allowed_pairs_ matrix
    if (BaseEncoding::is_ascii_base(x) && BaseEncoding::is_ascii_base(y)) {
        return allowed_pairs_[x][y];
    }

    // 3. Fall back to canonical (parent) bases
    const auto& enc = get_encoding();
    base_id cx = enc.get_canonical(x);
    base_id cy = enc.get_canonical(y);

    // If canonical is different, try the canonical pair
    if (cx != x || cy != y) {
        cx = BaseEncoding::to_lower(cx);
        cy = BaseEncoding::to_lower(cy);

        // Check extended pairs for canonical bases
        uint32_t ckey = (static_cast<uint32_t>(cx) << 16) | cy;
        auto cit = extended_pairs_.find(ckey);
        if (cit != extended_pairs_.end()) {
            return cit->second;
        }

        // Check allowed_pairs_ for canonical ASCII bases
        if (BaseEncoding::is_ascii_base(cx) && BaseEncoding::is_ascii_base(cy)) {
            return allowed_pairs_[cx][cy];
        }
    }

    return false;
}

bool
Fold::Options::
allow_paired_extended(const std::vector<base_id>& seq_ids, u_int32_t i, u_int32_t j) const
{
    std::tie(i, j) = std::minmax(i, j);
    return j-i>min_hairpin
        && this->allow_paired_extended(seq_ids[i-1], seq_ids[j-1])
        && (stru[i]==Options::ANY || stru[i]==Options::PAIRED_L || stru[i]==Options::PAIRED_LR || stru[i]==j)
        && (stru[j]==Options::ANY || stru[j]==Options::PAIRED_R || stru[j]==Options::PAIRED_LR || stru[j]==i);
}

auto
Fold::Options::
make_constraint_lin_extended(const std::string& seq, bool canonical_only /*=true*/) const
    -> std::tuple<std::unordered_map<base_id, std::vector<u_int32_t>>, std::vector<u_int32_t>, std::vector<bool>, std::vector<base_id>>
{
    const auto& enc = get_encoding();
    std::vector<base_id> seq_ids = enc.encode(seq);
    const auto L = seq_ids.size();

    if (stru.size() == 0)
        stru.resize(L+1, Options::ANY);

    for (auto i=L; i>=1; i--)
    {
        if (stru[i] > 0 && stru[i] <= L) // paired
            if ( (canonical_only && !this->allow_paired_extended(seq_ids[i-1], seq_ids[stru[i]-1])) || // delete non-canonical base-pairs
                    (stru[i] - i <= min_hairpin) ) // delete very short hairpin
                stru[i] = stru[stru[i]] = Options::UNPAIRED;
    }

    std::vector<bool> allow_unpaired_position(L+1, true);
    for (auto i=1u; i<=L; i++)
        allow_unpaired_position[i] = stru[i]==Options::ANY || stru[i]==Options::UNPAIRED;

    std::vector<u_int32_t> allow_unpaired_range(L+1, 0);
    auto firstpair = L+1;
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired_range[i] = firstpair;
        if (!allow_unpaired_position[i])
            firstpair = i;
    }

    // Build next_pair using hash map for unique bases in the sequence
    std::unordered_set<base_id> unique_bases = enc.get_unique_bases(seq_ids);
    std::unordered_map<base_id, std::vector<u_int32_t>> next_pair;

    for (auto nuc : unique_bases)
    {
        next_pair[nuc].resize(L+1, 0);
        u_int32_t next = 0;
        for (auto j=L; j>=1; j--)
        {
            next_pair[nuc][j] = next;
            if (stru[j] != Options::UNPAIRED && this->allow_paired_extended(seq_ids[j-1], nuc))
                next = j;
        }
    }

    return { next_pair, allow_unpaired_range, allow_unpaired_position, seq_ids };
}

auto
Fold::Options::
make_constraint(const std::string& seq, bool canonical_only /*=true*/) const
    -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>
{
    const auto L = seq.size();
    //std::vector<u_int32_t> stru(L+1, Options::ANY);
    //std::copy(std::begin(this->stru), std::end(this->stru), std::begin(stru));
    if (stru.size() == 0)
        stru.resize(L+1, Options::ANY);
    if (stru.size() < L+1)
        stru.resize(L+1, Options::ANY);

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

    return { allow_paired, allow_unpaired };
}

auto
Fold::Options::
make_constraint_extended(const std::string& seq, bool canonical_only /*=true*/) const
    -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>
{
    const auto& enc = get_encoding();
    std::vector<base_id> seq_ids = enc.encode(seq);
    const auto L = seq_ids.size();

    if (L == 0) {
        return { {}, {} };
    }

    if (stru.size() == 0)
        stru.resize(L+1, Options::ANY);

    if (stru.size() < L+1)
        stru.resize(L+1, Options::ANY);

    for (auto i=L; i>=1; i--)
        if (stru[i] > 0 && stru[i] <= L) // paired
            if ( (canonical_only && !this->allow_paired_extended(seq_ids[i-1], seq_ids[stru[i]-1])) ||
                    (stru[i] - i <= min_hairpin) )
                stru[i] = stru[stru[i]] = Options::UNPAIRED;

    std::vector<bool> pk(L+1, false);
    for (auto i=1u; i<=L; i++)
        if (stru[i] > 0 && stru[i] <= L) // paired
            for (auto k=i+1; k<stru[i]; k++)
                if (stru[k] <= L && stru[k] > stru[i]) // paired & pk
                    pk[i] = pk[stru[i]] = pk[k] = pk[stru[k]] = true;

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1, false));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1, false));
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired[i][i-1] = true;
        allow_unpaired[i][i] = stru[i]==Options::ANY || stru[i]==Options::UNPAIRED || pk[i];
        bool bp_l = stru[i]==Options::ANY || stru[i]==Options::PAIRED_L || stru[i]==Options::PAIRED_LR;
        for (auto j=i+1; j<=L; j++)
        {
            allow_paired[i][j] = j-i > min_hairpin;
            bool bp_r = stru[j]==Options::ANY || stru[j]==Options::PAIRED_R || stru[j]==Options::PAIRED_LR;
            allow_paired[i][j] = allow_paired[i][j] && ((bp_l && bp_r) || stru[i]==j);
            if (canonical_only)
                allow_paired[i][j] = allow_paired[i][j] && this->allow_paired_extended(seq_ids[i-1], seq_ids[j-1]);
            allow_unpaired[i][j] = allow_unpaired[i][j-1] && allow_unpaired[j][j];
        }
    }

    return { allow_paired, allow_unpaired };
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