#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include <memory>
#include "trimatrix.h"

class Fold
{
    public:
        struct Options
        {
            enum {
                UNPAIRED  =  0u, // 'x'
                ANY       = -1u, // '.'
                PAIRED_L  = -2u, // '<'
                PAIRED_R  = -3u, // '>'
                PAIRED_LR = -4u, // '|'
            };

            size_t min_hairpin;
            size_t max_internal;
            size_t max_helix;
            std::vector<u_int32_t> stru;
            std::vector<std::pair<u_int32_t, u_int32_t>> stru2;
            bool use_penalty;
            std::vector<u_int32_t> ref;
            std::vector<std::pair<u_int32_t,u_int32_t>> ref2;
            float pos_paired;
            float neg_paired;
            float pos_unpaired;
            float neg_unpaired;

            Options() : 
                min_hairpin(3),
                max_internal(30),
                max_helix(30),
                use_penalty(false)
            {    
            }

            Options& min_hairpin_loop_length(size_t s)
            {
                this->min_hairpin = s;
                return *this;
            }

            Options& max_internal_loop_length(size_t s)
            {
                this->max_internal = s;
                return *this;
            }

            Options& max_helix_length(size_t s)
            {
                this->max_helix = s;
                return *this;
            }

            Options& constraints(const std::vector<u_int32_t>& s)
            {
                this->stru = s;
                return *this;
            }

            Options& constraints(const std::vector<std::pair<u_int32_t, u_int32_t>>& s)
            {
                this->stru2 = s;
                return *this;
            }

            Options& penalty(const std::vector<u_int32_t>& ref, float pos_paired=0, float neg_paired=0, float pos_unpaired=0, float neg_unpaired=0)
            {
                this->use_penalty = pos_paired!=0 || neg_paired!=0 || pos_unpaired!=0 || neg_unpaired!=0;
                this->ref = ref;
                this->pos_paired = pos_paired;
                this->neg_paired = neg_paired;
                this->pos_unpaired = pos_unpaired;
                this->neg_unpaired = neg_unpaired;
                return *this;
            }

            Options& penalty(const std::vector<std::pair<u_int32_t, u_int32_t>>& ref2, 
                        float pos_paired=0, float neg_paired=0, float pos_unpaired=0, float neg_unpaired=0)
            {
                this->use_penalty = pos_paired!=0 || neg_paired!=0 || pos_unpaired!=0 || neg_unpaired!=0;
                this->ref2 = ref2;
                this->pos_paired = pos_paired;
                this->neg_paired = neg_paired;
                this->pos_unpaired = pos_unpaired;
                this->neg_unpaired = neg_unpaired;
                return *this;
            }

            auto make_constraint(const std::string& seq, bool canonical_only=true)
                -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>;
            auto make_penalty(size_t L)
                -> std::tuple<TriMatrix<float>, std::vector<std::vector<float>>, float>;
        };

    public:
        static bool allow_paired(char x, char y);
        static auto parse_paren(const std::string& paren) 
            -> std::vector<u_int32_t>;
        static auto make_paren(const std::vector<u_int32_t>& p) -> std::string;
};