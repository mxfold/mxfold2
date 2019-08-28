#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <regex>
#include <cstdlib>
#include <cassert>
#include "parameter.h"

using namespace std::literals::string_literals;

#include "default_params.h"

template <typename Itr1, typename Itr2>
static
void convert_sequence(Itr1 b1, Itr1 e1, Itr2 b2)
{
    for (auto it = b1; it != e1; ++it)
    {
        switch (tolower(*it)) {
            default:  *b2++ = 0; break;
            case 'a': *b2++ = 1; break;
            case 'c': *b2++ = 2; break;
            case 'g': *b2++ = 3; break;
            case 'u':
            case 't': *b2++ = 4; break;
        }
    }
}

template <typename S>
static
auto convert_sequence(const std::string& seq) -> S
{
    S converted_seq(seq.size());
    convert_sequence(std::begin(seq), std::end(seq), std::begin(converted_seq));
    return converted_seq;
}

static int pair[5][5] = {
   // _  A  C  G  U 
    { 0, 0, 0, 0, 0 }, // _
    { 0, 0, 0, 0, 5 }, // A
    { 0, 0, 0, 1, 0 }, // C
    { 0, 0, 2, 0, 3 }, // G
    { 0, 6, 0, 4, 0 }, // U
};

namespace detail
{
    template <typename T, size_t NDIMS> struct vector_builder
    {
        using type = std::vector<typename vector_builder<T, NDIMS-1>::type>;

        static type read_values(std::istream& is, std::vector<size_t> d, std::vector<size_t> s, std::vector<size_t> e)
        {
            const auto d0 = d.front(); d.erase(d.begin());
            const auto s0 = s.front(); s.erase(s.begin());
            const auto e0 = e.front(); e.erase(e.begin());
            type v = type(d0);
            for (auto i=s0; i!=d0-e0; i++)
                v[i] = vector_builder<T, NDIMS-1>::read_values(is, d, s, e);
            return v;
        }

        static type read_values(int*& val, std::vector<size_t> d, std::vector<size_t> s, std::vector<size_t> e)
        {
            const auto d0 = d.front(); d.erase(d.begin());
            const auto s0 = s.front(); s.erase(s.begin());
            const auto e0 = e.front(); e.erase(e.begin());
            type v = type(d0);
            for (auto i=s0; i!=d0-e0; i++)
                v[i] = vector_builder<T, NDIMS-1>::read_values(val, d, s, e);
            return v;
        }
    };

    template <typename T> struct vector_builder<T, 1>
    {
        using type = std::vector<T>;

        static type read_values(std::istream& is, std::vector<size_t> d, std::vector<size_t> s, std::vector<size_t> e)
        {
            const int DEF = -50;
            const int NST = 0;
            const int MY_INF = std::numeric_limits<int>::max();

            const auto d0 = d.front();
            const auto s0 = s.front();
            const auto e0 = e.front();

            std::vector<T> values(d0, 0);
            size_t i = s0;
            while (is && i < d0-e0)
            {
                std::string line;
                if (!std::getline(is, line)) break;
                line = std::regex_replace(line, std::regex("/\\*.*\\*/"), ""); // remove comment
                std::istringstream ss(line);
                std::string v;
                while (ss >> v && i < d0-e0)
                {
                    if (v == "INF")
                        values[i++] = MY_INF;
                    else if (v == "DEF")
                        values[i++] = DEF;
                    else if (v == "NST")
                        values[i++] = NST;
                    else
                        values[i++] = atoi(v.c_str());
                }
            }

            return values;
        }

        static type read_values(int*& val, std::vector<size_t> d, std::vector<size_t> s, std::vector<size_t> e)
        {
            const auto d0 = d.front();
            const auto s0 = s.front();
            const auto e0 = e.front();

            std::vector<T> values(d0, 0);
            size_t i = s0;
            while (i < d0-e0)
                values[i++] = *val++;

            return values;
        }
    };
};

template <typename T = int, typename... SIZE_T>
auto read_values(std::istream& is, SIZE_T... dims)
{
    std::vector<size_t> vec_dims{dims...};
    constexpr size_t N = sizeof...(dims) / 3;
    std::vector<size_t> d(N), s(N), e(N);
    std::copy(vec_dims.begin(), vec_dims.begin()+N, d.begin());
    std::copy(vec_dims.begin()+N, vec_dims.begin()+N*2, s.begin());
    std::copy(vec_dims.begin()+N*2, vec_dims.end(), e.begin());
    return detail::vector_builder<T, N>::read_values(is, d, s, e);
}

template <typename T = int, typename... SIZE_T>
auto read_values(int*& v, SIZE_T... dims)
{
    std::vector<size_t> vec_dims{dims...};
    constexpr size_t N = sizeof...(dims) / 3;
    std::vector<size_t> d(N), s(N), e(N);
    std::copy(vec_dims.begin(), vec_dims.begin()+N, d.begin());
    std::copy(vec_dims.begin()+N, vec_dims.begin()+N*2, s.begin());
    std::copy(vec_dims.begin()+N*2, vec_dims.end(), e.begin());
    return detail::vector_builder<T, N>::read_values(v, d, s, e);
}

bool
MFE::
load(const char* filename)
{
    const auto NBPAIRS = 7;
    std::ifstream is(filename);
    if (!is.is_open()) return false;
    std::string line;
    while (std::getline(is, line)) 
    {
        if (std::regex_search(line, std::regex("^##"))) // header
            continue;
        std::smatch m;
        if (std::regex_match(line, m, std::regex("^# (.+)$")))
        {
            if (m[1] == "stack")
                stack_ = read_values(is, NBPAIRS+1u, NBPAIRS+1u, 1u, 1u, 0u, 0u);
            else if (m[1] == "stack_enthalpies")
                /* stack_dH_  =*/ read_values(is, NBPAIRS+1u, NBPAIRS+1u, 1u, 1u, 0u, 0u);
            else if (m[1] == "hairpin")
                hairpin_ = read_values(is, 31u, 0u, 0u);
            else if (m[1] == "hairpin_enthalpies")
                /* hairpin_dH_ = */ read_values(is, 31u, 0u, 0u);
            else if (m[1] == "bulge")
                bulge_ = read_values(is, 31u, 0u, 0u);
            else if (m[1] == "bulge_enthalpies")
                /* bulge_dH_ = */ read_values(is, 31u, 0u, 0u);
            else if (m[1] == "interior")
                internal_ = read_values(is, 31u, 0u, 0u);
            else if (m[1] == "interior_enthalpies")
                /* internal_dH_ = */ read_values(is, 31u, 0u, 0u);
            else if (m[1] == "mismatch_exterior")
                mismatch_external_ = read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_exterior_enthalpies")
                /* mismatch_exterior_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_hairpin")
                mismatch_hairpin_ = read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_hairpin_enthalpies")
                /* mismatch_hairpin_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_interior")
                mismatch_internal_ = read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_interior_enthalpies")
                /* mismatch_internal_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_interior_1n")
                mismatch_internal_1n_ = read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_interior_1n_enthalpies")
                /* mismatch_internal_1n_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_interior_23")
                mismatch_internal_23_ = read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_interior_23_enthalpies")
                /*  mismatch_internal_23_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_multi")
                mismatch_multi_ = read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "mismatch_multi_enthalpies")
                /* mismatch_multi_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "int11")
                int11_ = read_values(is, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "int11_enthalpies")
                /* int11_dH_ = */ read_values(is, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "int21")
                int21_ = read_values(is, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "int21_enthalpies")
                /* int21_dH_ = */ read_values(is, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
            else if (m[1] == "int22")
                int22_ = read_values(is, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 5u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u);
            else if (m[1] == "int22_enthalpies")
                /* int22_dH_ = */ read_values(is, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 5u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u);
            else if (m[1] == "dangle5")
                dangle5_ = read_values(is, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);
            else if (m[1] == "dangle5_enthalpies")
                /* dangle5_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);
            else if (m[1] == "dangle3")
                dangle3_ = read_values(is, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);
            else if (m[1] == "dangle3_enthalpies")
                /* dangle3_dH_ = */ read_values(is, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);
            else if (m[1] == "ML_params")
            {
                auto v = read_values(is, 6u, 0u, 0u);
                ml_base_ = v[0];
                // ml_base_dH_ = v[1];
                ml_closing_ = v[2];
                // ml_closing_dH_ = v[3];
                ml_intern_ = v[4];
                //ml_intern_dH_ = v[5];
            }
            else if (m[1] == "NINIO")
            {
                auto v = read_values(is, 3u, 0u, 0u);
                ninio_ = v[0];
                // ninio_dH_ = v[1];
                max_ninio_ = v[2];
            }
            else if (m[1] == "Triloops" || m[1] == "Tetraloops" || m[1] == "Hexaloops")
            {
                while (std::getline(is, line)) 
                {
                    std::smatch m;
                    if (!std::regex_match(line, m, std::regex("^([ACGU]+)\\s+(-?\\d+)\\s+(-?\\d+)\\s*$")))
                        break;
                    special_loops_.emplace(::convert_sequence<SeqType>(m[1]), atoi(m[2].str().c_str()));
                    // special_loops_dH_.emplace(m[1], atoi(m[3]);
                }
            }
            else if (m[1]  == "Misc")
            {
                auto v = read_values(is, 4u, 0u, 0u);
                duplex_init_ = v[0];
                // duplex_init_dH_ = v[1];
                terminalAU_ = v[2];
                // terminalAU_dH_ = v[3];
            }
            else if (m[1]  == "END")
                break;
            else
                return false;
        }
    }
    return true;
}

bool
MFE::
load_default()
{
    const auto NBPAIRS = 7;
#if 0
    const int DEF = -50;
    const int NST = 0;
#ifdef INF
#undef INF
#endif
    const int INF = std::numeric_limits<int>::max();
#endif
#include "default_params.h"
    int* values = default_params;

    // stack
    stack_ = read_values(values, NBPAIRS+1u, NBPAIRS+1u, 1u, 1u, 0u, 0u);
    /* stack_dH_  =*/ read_values(values, NBPAIRS+1u, NBPAIRS+1u, 1u, 1u, 0u, 0u);

    // mismatch_hairpin
    mismatch_hairpin_ = read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
    /* mismatch_hairpin_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);

    // mismatch_interior
    mismatch_internal_ = read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
    /* mismatch_internal_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);

    // mismatch_interior_1n
    mismatch_internal_1n_ = read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
    /* mismatch_internal_1n_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);

    // mismatch_interior_23
    mismatch_internal_23_ = read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
    /*  mismatch_internal_23_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);

    // "mismatch_multi
    mismatch_multi_ = read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
    /* mismatch_multi_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);

    // mismatch_exterior
    mismatch_external_ = read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);
    /* mismatch_exterior_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 5u, 1u, 0u, 0u, 0u, 0u, 0u);

    // dangle5
    dangle5_ = read_values(values, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);
    /* dangle5_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);

    // dangle3
    dangle3_ = read_values(values, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);
    /* dangle3_dH_ = */ read_values(values, NBPAIRS+1u, 5u, 1u, 0u, 0u, 0u);

    // int11
    int11_ = read_values(values, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u);
    /* int11_dH_ = */ read_values(values, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u);

    // int21
    int21_ = read_values(values, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    /* int21_dH_ = */ read_values(values, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    // int22
    int22_ = read_values(values, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 5u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u);
    /* int22_dH_ = */ read_values(values, NBPAIRS+1u, NBPAIRS+1u, 5u, 5u, 5u, 5u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u);

    // hairpin
    hairpin_ = read_values(values, 31u, 0u, 0u);
    /* hairpin_dH_ = */ read_values(values, 31u, 0u, 0u);

    // bulge
    bulge_ = read_values(values, 31u, 0u, 0u);
    /* bulge_dH_ = */ read_values(values, 31u, 0u, 0u);

    // interior
    internal_ = read_values(values, 31u, 0u, 0u);
    /* internal_dH_ = */ read_values(values, 31u, 0u, 0u);

    // ML_params
    {
        auto v = read_values(values, 6u, 0u, 0u);
        ml_base_ = v[0];
        // ml_base_dH_ = v[1];
        ml_closing_ = v[2];
        // ml_closing_dH_ = v[3];
        ml_intern_ = v[4];
        //ml_intern_dH_ = v[5];
    }
    // NINIO
    {
        auto v = read_values(values, 3u, 0u, 0u);
        ninio_ = v[0];
        // ninio_dH_ = v[1];
        max_ninio_ = v[2];
    }

    // Misc
    {
        auto v = read_values(values, 4u, 0u, 0u);
        duplex_init_ = v[0];
        // duplex_init_dH_ = v[1];
        terminalAU_ = v[2];
        // terminalAU_dH_ = v[3];
    }

    // Triloops, Tetraloops, Hexaloops
    for (auto i = 0; default_params_sl[i].rna !=NULL; i++)
    {
        special_loops_.emplace(::convert_sequence<SeqType>(default_params_sl[i].rna), default_params_sl[i].e);
    }
    
    return true;
}

auto
MFE::
convert_sequence(const std::string& seq) -> SeqType
{
    const auto L = seq.size();
    SeqType converted_seq(L+2);
    ::convert_sequence(std::begin(seq), std::end(seq), &converted_seq[1]);
    converted_seq[0] = converted_seq[L];
    converted_seq[L+1] = converted_seq[1];

    return converted_seq;
}

template <typename R>
R
MFE::
hairpin(const SeqType& s, size_t i, size_t j)
{
    const auto l = (j-1)-(i+1)+1;
    auto e = 0;

    if (l <= 30)
        e += hairpin_[l];
    else
        e += hairpin_[30] + (int)(lxc_ * log(l / 30.));

    if (l < 3) return e / -100.;

    if (3 <= l && l <= 6) {
        SeqType sl(&s[i], &s[j]+1);
        auto it = special_loops_.find(sl);
        if (it != std::end(special_loops_))
            return it->second / -100.;
    }           
    
    const auto type = ::pair[s[i]][s[j]];
    if (l == 3)
        e += type > 2 ? terminalAU_ : 0;
    else
        e += mismatch_hairpin_[type][s[i+1]][s[j-1]];

    return e / -100.;
}

template
float
MFE::
hairpin(const SeqType& s, size_t i, size_t j);

template <typename R>
R
MFE::
single_loop(const SeqType& s, size_t i, size_t j, size_t k, size_t l)
{
    const auto type1 = ::pair[s[i]][s[j]];
    const auto type2 = ::pair[s[l]][s[k]];
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);
    int e = std::numeric_limits<int>::max();

    if (ll==0) // stack
        e = stack_[type1][type2];
    else if (ls==0) // bulge
    {
        e = ll<=30 ? bulge_[ll] : bulge_[30] + (int)(lxc_ * log(ll / 30.));
        if (ll==1) 
            e += stack_[type1][type2];
        else
        {
            if (type1 > 2)
                e += terminalAU_;
            if (type2 > 2)
                e += terminalAU_;
        }
    }
    else // internal loop
    {
        if (ll==1 && ls==1) // 1x1 loop
            e = int11_[type1][type2][s[i+1]][s[j-1]];
        else if (l1==2 && l2==1) // 2x1 loop
            e = int21_[type2][type1][s[l+1]][s[i+1]][s[k-1]];
        else if (l1==1 && l2==2) // 1x2 loop
            e = int21_[type1][type2][s[i+1]][s[l+1]][s[j-1]];
        else if (ls==1) // 1xn loop
        {
            e = ll+1 <= 30 ? internal_[ll+1] : internal_[30] + (int)(lxc_ * log((ll+1) / 30.));
            e += std::min(max_ninio_, (int)(ll-ls) * ninio_);
            e += mismatch_internal_1n_[type1][s[i+1]][s[j-1]] + mismatch_internal_1n_[type2][s[l+1]][s[k-1]];
        }
        else if (ls==2 && ll==2) // 2x2 loop
            e = int22_[type1][type2][s[i+1]][s[k-1]][s[l+1]][s[j-1]];
        else if (ls==2 && ll==3) // 2x3 loop
        {
            e = internal_[ls+ll] + ninio_;
            e += mismatch_internal_23_[type1][s[i+1]][s[j-1]] + mismatch_internal_23_[type2][s[l+1]][s[k-1]];
        }
        else // generic internal loop
        {
            e = ls+ll <= 30 ? internal_[ls+ll] : internal_[30] + (int)(lxc_ * log((ls+ll) / 30.));
            e += std::min(max_ninio_, (int)(ll-ls) * ninio_);
            e += mismatch_internal_[type1][s[i+1]][s[j-1]] + mismatch_internal_[type2][s[l+1]][s[k-1]];
        }
    }

    return e / -100.;
}

template
float
MFE::
single_loop(const SeqType& s, size_t i, size_t j, size_t k, size_t l);

template <typename R>
R
MFE::
multi_loop(const SeqType& s, size_t i, size_t j) 
{
    int e = 0;
    const auto type = ::pair[s[j]][s[i]];
    e += mismatch_multi_[type][s[j-1]][s[i+1]];
    if (type > 2) 
        e += terminalAU_;
    e += ml_intern_;
    e += ml_closing_;

    return e / -100.;
}
template
float
MFE::
multi_loop(const SeqType& s, size_t i, size_t j);

template <typename R>
R
MFE::
multi_paired(const SeqType& s, size_t i, size_t j)
{
    const auto L = s.size()-2;
    int e = 0;
    const auto type = ::pair[s[i]][s[j]];
    if (i-1>=1 && j+1<=L)
        e += mismatch_multi_[type][s[i-1]][s[j+1]];
    else if (i-1>=1)
        e += dangle5_[type][s[i-1]];
    else if (j+1<=L)
        e += dangle3_[type][s[j+1]];
    if (type > 2) 
        e += terminalAU_;
    e += ml_intern_;

    return e / -100.;
}

template
float
MFE::
multi_paired(const SeqType& s, size_t i, size_t j);

template <typename R>
R
MFE::
multi_unpaired(const SeqType& s, size_t i)
{
    const auto e = ml_base_;

    return e / -100.;
}

template
float
MFE::
multi_unpaired(const SeqType& s, size_t i);

template <typename R>
R
MFE::
external_paired(const SeqType& s, size_t i, size_t j)
{
    const auto L = s.size()-2;
    auto e = 0.;
    const auto type = ::pair[s[i]][s[j]];
    if (i-1>=1 && j+1<=L)
        e += mismatch_external_[type][s[i-1]][s[j+1]];
    else if (i-1>=1)
        e += dangle5_[type][s[i-1]];
    else if (j+1<=L)
        e += dangle3_[type][s[j+1]];
    if (type > 2) 
        e += terminalAU_;
    
    return -e / 100.;
}

template
float
MFE::
external_paired(const SeqType& s, size_t i, size_t j);


#define NBASES 4
#define NBPAIRS 7

MFETorch::
MFETorch() :
    stack_(register_parameter("stack", torch::zeros({NBPAIRS+1, NBPAIRS+1u}, torch::dtype(torch::kFloat)))),
    hairpin_(register_parameter("hairpin", torch::zeros({31}, torch::dtype(torch::kFloat)))),
    bulge_(register_parameter("bulge", torch::zeros({31}, torch::dtype(torch::kFloat)))),
    internal_(register_parameter("internal", torch::zeros({31}, torch::dtype(torch::kFloat)))),
    mismatch_external_(register_parameter("mismatch_external", torch::zeros({NBPAIRS+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    mismatch_hairpin_(register_parameter("mismatch_hairpin", torch::zeros({NBPAIRS+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    mismatch_internal_(register_parameter("mismatch_internal", torch::zeros({NBPAIRS+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    mismatch_internal_1n_(register_parameter("mismatch_internal_1n", torch::zeros({NBPAIRS+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    mismatch_internal_23_(register_parameter("mismatch_internal_23", torch::zeros({NBPAIRS+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    mismatch_multi_(register_parameter("mismatch_multi", torch::zeros({NBPAIRS+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    int11_(register_parameter("int11", torch::zeros({NBPAIRS+1, NBPAIRS+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    int21_(register_parameter("int21", torch::zeros({NBPAIRS+1, NBPAIRS+1, NBASES+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    int22_(register_parameter("int22", torch::zeros({NBPAIRS+1, NBPAIRS+1, NBASES+1, NBASES+1, NBASES+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    dangle5_(register_parameter("dangle5", torch::zeros({NBPAIRS+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    dangle3_(register_parameter("dangle3", torch::zeros({NBPAIRS+1, NBASES+1}, torch::dtype(torch::kFloat)))),
    ml_base_(register_parameter("ml_base", torch::zeros({1}))),
    ml_closing_(register_parameter("ml_closing", torch::zeros({1}, torch::dtype(torch::kFloat)))),
    ml_intern_(register_parameter("ml_intern", torch::zeros({1}, torch::dtype(torch::kFloat)))),
    ninio_(register_parameter("ninio", torch::zeros({1}, torch::dtype(torch::kFloat)))),
    max_ninio_(register_parameter("max_ninio", torch::zeros({1}, torch::dtype(torch::kFloat)))),
    duplex_init_(register_parameter("duplex_init", torch::zeros({1}, torch::dtype(torch::kFloat)))),
    terminalAU_(register_parameter("terminalAU", torch::zeros({1}, torch::dtype(torch::kFloat)))),
    lxc_(torch::full({1}, 107.856, torch::dtype(torch::kFloat)))
{

}

void
MFETorch::
init_accessors()
{
    stack_a_ = stack_.accessor<float, 2>();
    hairpin_a_ = hairpin_.accessor<float, 1>();
    bulge_a_ = bulge_.accessor<float, 1>();
    internal_a_ = internal_.accessor<float, 1>();
    mismatch_external_a_ = mismatch_external_.accessor<float, 3>();
    mismatch_hairpin_a_ = mismatch_hairpin_.accessor<float, 3>();
    mismatch_internal_a_ = mismatch_internal_.accessor<float, 3>();
    mismatch_internal_1n_a_ = mismatch_internal_1n_.accessor<float, 3>();
    mismatch_internal_23_a_ = mismatch_internal_23_.accessor<float, 3>();
    mismatch_multi_a_ = mismatch_multi_.accessor<float, 3>();
    int11_a_ = int11_.accessor<float, 4>();
    int21_a_ = int21_.accessor<float, 5>();
    int22_a_ = int22_.accessor<float, 6>();
    dangle5_a_ = dangle5_.accessor<float, 2>();
    dangle3_a_ = dangle3_.accessor<float, 2>();
    ml_base_a_ = ml_base_.accessor<float, 1>();
    ml_closing_a_ = ml_closing_.accessor<float, 1>();
    ml_intern_a_ = ml_intern_.accessor<float, 1>();
    ninio_a_ = ninio_.accessor<float, 1>();
    max_ninio_a_ = max_ninio_.accessor<float, 1>();
    duplex_init_a_ = duplex_init_.accessor<float, 1>();
    terminalAU_a_ = terminalAU_.accessor<float, 1>();
    lxc_a_ = lxc_.accessor<float, 1>();
}

auto
MFETorch::
convert_sequence(const std::string& seq) -> SeqType
{
    init_accessors();

    const auto L = seq.size();
    SeqType converted_seq(L+2);
    ::convert_sequence(std::begin(seq), std::end(seq), &converted_seq[1]);
    converted_seq[0] = converted_seq[L];
    converted_seq[L+1] = converted_seq[1];

    return converted_seq;
}

bool
MFETorch::
load_default()
{
    torch::NoGradGuard no_grad;

    //const auto NBPAIRS = 7;
    int32_t* values = default_params;

    // stack
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=1; i1!=NBPAIRS+1; i1++)
            stack_[i0][i1] = *values++;
    values += NBPAIRS*NBPAIRS;
    
    // mismatch_hairpin
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                mismatch_hairpin_[i0][i1][i2] = *values++;
    values += NBPAIRS*(NBASES+1)*(NBASES+1);

    // mismatch_interior
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                mismatch_internal_[i0][i1][i2] = *values++;
    values += NBPAIRS*(NBASES+1)*(NBASES+1);

    // mismatch_interior_1n
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                mismatch_internal_1n_[i0][i1][i2] = *values++;
    values += NBPAIRS*(NBASES+1)*(NBASES+1);

    // mismatch_interior_23
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                mismatch_internal_23_[i0][i1][i2] = *values++;
    values += NBPAIRS*(NBASES+1)*(NBASES+1);

    // mismatch_multi
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                mismatch_multi_[i0][i1][i2] = *values++;
    values += NBPAIRS*(NBASES+1)*(NBASES+1);

    // mismatch_exterior
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                mismatch_external_[i0][i1][i2] = *values++;
    values += NBPAIRS*(NBASES+1)*(NBASES+1);

    // dangle5
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            dangle5_[i0][i1] = *values++;
    values += NBPAIRS*(NBASES+1);

    // dangle3
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=0; i1!=NBASES+1; i1++)
            dangle3_[i0][i1] = *values++;
    values += NBPAIRS*(NBASES+1);

    // int11
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=1; i1!=NBPAIRS+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                for (auto i3=0; i3!=NBASES+1; i3++)
                    int11_[i0][i1][i2][i3] = *values++;
    values += NBPAIRS*NBPAIRS*(NBASES+1)*(NBASES+1);

    // int21
    for (auto i0=1; i0!=NBPAIRS+1; i0++)
        for (auto i1=1; i1!=NBPAIRS+1; i1++)
            for (auto i2=0; i2!=NBASES+1; i2++)
                for (auto i3=0; i3!=NBASES+1; i3++)
                    for (auto i4=0; i4!=NBASES+1; i4++)
                        int21_[i0][i1][i2][i3][i4] = *values++;
    values += NBPAIRS*NBPAIRS*(NBASES+1)*(NBASES+1)*(NBASES+1);

    // int22
    for (auto i0=1; i0!=NBPAIRS; i0++)
        for (auto i1=1; i1!=NBPAIRS; i1++)
            for (auto i2=1; i2!=NBASES+1; i2++)
                for (auto i3=1; i3!=NBASES+1; i3++)
                    for (auto i4=1; i4!=NBASES+1; i4++)
                        for (auto i5=1; i5!=NBASES+1; i5++)
                        int22_[i0][i1][i2][i3][i4][i5] = *values++;
    values += (NBPAIRS-1)*(NBPAIRS-1)*NBASES*NBASES*NBASES*NBASES;

    // hairpin
    for (auto i0=0; i0!=31; i0++)
        hairpin_[i0] = *values++;
    values += 31;

    // bulge
    for (auto i0=0; i0!=31; i0++)
        bulge_[i0] = *values++;
    values += 31;

    // interior
    for (auto i0=0; i0!=31; i0++)
        internal_[i0] = *values++;
    values += 31;

    // ML_params
    ml_base_[0] = *values++; values++;
    ml_closing_[0] = *values++; values++;
    ml_intern_[0] = *values++; values++;

    // NINIO
    ninio_[0] = *values++; values++;
    max_ninio_[0] = *values++;

    // Misc
    duplex_init_[0] = *values++; values++;
    terminalAU_[0] = *values++; values++;

    return true;
}

template <>
torch::Tensor
MFETorch::
hairpin(const SeqType& s, size_t i, size_t j)
{
    const auto l = (j-1)-(i+1)+1;
    auto e = torch::zeros({}, torch::dtype(torch::kFloat));

    e += l<=30 ? hairpin_[l] : hairpin_[30] + lxc_[0] * log(l / 30.);

    if (l < 3) return e;

#if 0 // todo: special loops
    if (3 <= l && l <= 6) {
        SeqType sl(&s[i], &s[j]+l);
        auto it = special_loops_.find(sl);
        if (it != std::end(special_loops_))
            return it->second;
    }           
#endif

    const auto type = ::pair[s[i]][s[j]];
    if (l == 3)
    {
        if (type > 2)
            e += terminalAU_[0];
    }
    else
        e += mismatch_hairpin_[type][s[i+1]][s[j-1]];

    return -e / 100.;
}

template <>
float
MFETorch::
hairpin(const SeqType& s, size_t i, size_t j)
{
    const auto l = (j-1)-(i+1)+1;
    float e = 0.;

    e += l<=30 ? hairpin_a_.value()[l] : hairpin_a_.value()[30] + lxc_a_.value()[0] * log(l / 30.);

    if (l < 3) return e;

#if 0 // todo: special loops
    if (3 <= l && l <= 6) {
        SeqType sl(&s[i], &s[j]+l);
        auto it = special_loops_.find(sl);
        if (it != std::end(special_loops_))
            return it->second;
    }           
#endif

    const auto type = ::pair[s[i]][s[j]];
    if (l == 3)
    {
        if (type > 2)
            e += terminalAU_a_.value()[0];
    }
    else
        e += mismatch_hairpin_a_.value()[type][s[i+1]][s[j-1]];

    return -e / 100.;
}

template <>
torch::Tensor
MFETorch::
single_loop(const SeqType& s, size_t i, size_t j, size_t k, size_t l)
{
    const auto type1 = ::pair[s[i]][s[j]];
    const auto type2 = ::pair[s[l]][s[k]];
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);
    auto e = torch::zeros({}, torch::dtype(torch::kFloat));

    if (ll==0) // stack
        e += stack_[type1][type2];
    else if (ls==0) // bulge
    {
        e += ll<=30 ? bulge_[ll] : bulge_[30] + lxc_[0] * log(ll / 30.);
        if (ll==1)
            e += stack_[type1][type2];
        else
        {
            if (type1 > 2)
                e += terminalAU_[0];
            if (type2 > 2)
                e += terminalAU_[0];
        }
    }
    else if (ll==1 && ls==1) // 1x1 loop
        e += int11_[type1][type2][s[i+1]][s[j-1]];
    else if (l1==2 && l2==1) // 2x1 loop
        e += int21_[type2][type1][s[l+1]][s[i+1]][s[k-1]];
    else if (l1==1 && l2==2) // 1x2 loop
        e += int21_[type1][type2][s[i+1]][s[l+1]][s[j-1]];
    else if (ls==1) // 1xn loop
    {
        e += ll+1 <= 30 ? internal_[ll+1] : internal_[30] + lxc_[0] * log((ll+1) / 30.);
        e += torch::min(max_ninio_[0], (int)(ll-ls) * ninio_[0]);
        e += mismatch_internal_1n_[type1][s[i+1]][s[j-1]] + mismatch_internal_1n_[type2][s[l+1]][s[k-1]];
    }
    else if (ls==2 && ll==2) // 2x2 loop
        e += int22_[type1][type2][s[i+1]][s[k-1]][s[l+1]][s[j-1]];
    else if (ls==2 && ll==3) // 2x3 loop
    {
        e += internal_[ls+ll] + ninio_[0];
        e += mismatch_internal_23_[type1][s[i+1]][s[j-1]] + mismatch_internal_23_[type2][s[l+1]][s[k-1]];
    }
    else // generic internal loop
    {
        e += ls+ll <= 30 ? internal_[ls+ll] : internal_[30] + lxc_[0] * log((ls+ll) / 30.);
        e += torch::min(max_ninio_[0], (int)(ll-ls) * ninio_[0]);
        e += mismatch_internal_[type1][s[i+1]][s[j-1]] + mismatch_internal_[type2][s[l+1]][s[k-1]];
    }
    return -e / 100.;
}

template <>
float
MFETorch::
single_loop(const SeqType& s, size_t i, size_t j, size_t k, size_t l)
{
    const auto type1 = ::pair[s[i]][s[j]];
    const auto type2 = ::pair[s[l]][s[k]];
    const auto l1 = (k-1)-(i+1)+1;
    const auto l2 = (j-1)-(l+1)+1;
    const auto [ls, ll] = std::minmax(l1, l2);
    float e = 0.;

    if (ll==0) // stack
        e += stack_a_.value()[type1][type2];
    else if (ls==0) // bulge
    {
        e += ll<=30 ? bulge_a_.value()[ll] : bulge_a_.value()[30] + lxc_a_.value()[0] * log(ll / 30.);
        if (ll==1)
            e += stack_a_.value()[type1][type2];
        else
        {
            if (type1 > 2)
                e += terminalAU_a_.value()[0];
            if (type2 > 2)
                e += terminalAU_a_.value()[0];
        }
    }
    else if (ll==1 && ls==1) // 1x1 loop
        e += int11_a_.value()[type1][type2][s[i+1]][s[j-1]];
    else if (l1==2 && l2==1) // 2x1 loop
        e += int21_a_.value()[type2][type1][s[l+1]][s[i+1]][s[k-1]];
    else if (l1==1 && l2==2) // 1x2 loop
        e += int21_a_.value()[type1][type2][s[i+1]][s[l+1]][s[j-1]];
    else if (ls==1) // 1xn loop
    {
        e += ll+1 <= 30 ? internal_a_.value()[ll+1] : internal_a_.value()[30] + lxc_a_.value()[0] * log((ll+1) / 30.);
        e += std::min(max_ninio_a_.value()[0], (int)(ll-ls) * ninio_a_.value()[0]);
        e += mismatch_internal_1n_a_.value()[type1][s[i+1]][s[j-1]] + mismatch_internal_1n_a_.value()[type2][s[l+1]][s[k-1]];
    }
    else if (ls==2 && ll==2) // 2x2 loop
        e += int22_a_.value()[type1][type2][s[i+1]][s[k-1]][s[l+1]][s[j-1]];
    else if (ls==2 && ll==3) // 2x3 loop
    {
        e += internal_a_.value()[ls+ll] + ninio_a_.value()[0];
        e += mismatch_internal_23_a_.value()[type1][s[i+1]][s[j-1]] + mismatch_internal_23_a_.value()[type2][s[l+1]][s[k-1]];
    }
    else // generic internal loop
    {
        e += ls+ll <= 30 ? internal_a_.value()[ls+ll] : internal_a_.value()[30] + lxc_a_.value()[0] * log((ls+ll) / 30.);
        e += std::min(max_ninio_a_.value()[0], (int)(ll-ls) * ninio_a_.value()[0]);
        e += mismatch_internal_a_.value()[type1][s[i+1]][s[j-1]] + mismatch_internal_a_.value()[type2][s[l+1]][s[k-1]];
    }
    return -e / 100.;
}

template <>
torch::Tensor
MFETorch::
multi_loop(const SeqType& s, size_t i, size_t j)
{
    auto e = torch::zeros({}, torch::dtype(torch::kFloat));
    const auto type = ::pair[s[j]][s[i]];
    e += mismatch_multi_[type][s[j-1]][s[i+1]];
    if (type > 2) 
        e += terminalAU_[0];
    e += ml_intern_[0];
    e += ml_closing_[0];

    return -e / 100.;
}

template <>
float
MFETorch::
multi_loop(const SeqType& s, size_t i, size_t j)
{
    float e = 0.;
    const auto type = ::pair[s[j]][s[i]];
    e += mismatch_multi_a_.value()[type][s[j-1]][s[i+1]];
    if (type > 2) 
        e += terminalAU_a_.value()[0];
    e += ml_intern_a_.value()[0];
    e += ml_closing_a_.value()[0];

    return -e / 100.;
}

template <>
torch::Tensor
MFETorch::
multi_paired(const SeqType& s, size_t i, size_t j)
{
    const auto L = s.size()-2;
    auto e = torch::zeros({}, torch::dtype(torch::kFloat));
    const auto type = ::pair[s[i]][s[j]];
    if (i-1>=1 && j+1<=L)
        e += mismatch_multi_[type][s[i-1]][s[j+1]];
    else if (i-1>=1)
        e += dangle5_[type][s[i-1]];
    else if (j+1<=L)
        e += dangle3_[type][s[j+1]];
    if (type > 2) 
        e += terminalAU_[0];
    e += ml_intern_[0];

    return -e / 100.;
}

template <>
float
MFETorch::
multi_paired(const SeqType& s, size_t i, size_t j)
{
    const auto L = s.size()-2;
    float e = 0.;
    const auto type = ::pair[s[i]][s[j]];
    if (i-1>=1 && j+1<=L)
        e += mismatch_multi_a_.value()[type][s[i-1]][s[j+1]];
    else if (i-1>=1)
        e += dangle5_a_.value()[type][s[i-1]];
    else if (j+1<=L)
        e += dangle3_a_.value()[type][s[j+1]];
    if (type > 2) 
        e += terminalAU_a_.value()[0];
    e += ml_intern_a_.value()[0];

    return -e / 100.;
}

template <>
torch::Tensor
MFETorch::
multi_unpaired(const SeqType& s, size_t i)
{
    return -ml_base_[0] / 100.;
}

template <>
float
MFETorch::
multi_unpaired(const SeqType& s, size_t i)
{
    return -ml_base_a_.value()[0] / 100.;
}

template <>
torch::Tensor
MFETorch::
external_zero(const SeqType& seq)
{
    return torch::zeros({}, torch::dtype(torch::kFloat)); 
}

template <>
float
MFETorch::
external_zero(const SeqType& seq)
{
    return 0.;
}

template <>
torch::Tensor
MFETorch::
external_paired(const SeqType& s, size_t i, size_t j)
{
    const auto L = s.size()-2;
    auto e = torch::zeros({}, torch::dtype(torch::kFloat));
    const auto type = ::pair[s[i]][s[j]];
    if (i-1>=1 && j+1<=L)
        e += mismatch_external_[type][s[i-1]][s[j+1]];
    else if (i-1>=1)
        e += dangle5_[type][s[i-1]];
    else if (j+1<=L)
        e += dangle3_[type][s[j+1]];
    if (type > 2) 
        e += terminalAU_[0];
    
    return -e / 100.;
}

template <>
float
MFETorch::
external_paired(const SeqType& s, size_t i, size_t j)
{
    const auto L = s.size()-2;
    float e = 0.;
    const auto type = ::pair[s[i]][s[j]];
    if (i-1>=1 && j+1<=L)
        e += mismatch_external_a_.value()[type][s[i-1]][s[j+1]];
    else if (i-1>=1)
        e += dangle5_a_.value()[type][s[i-1]];
    else if (j+1<=L)
        e += dangle3_a_.value()[type][s[j+1]];
    if (type > 2) 
        e += terminalAU_a_.value()[0];
    
    return -e / 100.;
}

template <>
torch::Tensor
MFETorch::
external_unpaired(const SeqType& seq, size_t i)
{
    return torch::zeros({}, torch::dtype(torch::kFloat));
}

template <>
float
MFETorch::
external_unpaired(const SeqType& seq, size_t i)
{
    return 0.;
}
