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

            //size_t seq_len;
            size_t min_hairpin;
            size_t max_internal;
            size_t max_helix;
            std::vector<u_int32_t> stru;
            std::vector<std::pair<u_int32_t, u_int32_t>> stru2;
            bool use_margin;
            std::vector<u_int32_t> ref;
            std::vector<std::pair<u_int32_t,u_int32_t>> ref2;
            float pos_paired;
            float neg_paired;
            float pos_unpaired;
            float neg_unpaired;
            std::vector<float> score_paired_position_;
            std::vector<std::vector<bool>> allowed_pairs_;

            Options() : 
                min_hairpin(3),
                max_internal(30),
                max_helix(30),
                pos_paired(0), neg_paired(0),
                pos_unpaired(0), neg_unpaired(0),
                use_margin(false),
                allowed_pairs_(256, std::vector<bool>(256, false))
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
#if 0
            Options& constraints(const std::vector<std::pair<u_int32_t, u_int32_t>>& s)
            {
                throw "this call has been obsolete.";
                this->stru2 = s;
                return *this;
            }
#endif
            Options& margin_terms(const std::vector<u_int32_t>& ref,
                        float pos_paired=0, float neg_paired=0, float pos_unpaired=0, float neg_unpaired=0)
            {
                this->use_margin = pos_paired!=0 || neg_paired!=0 || pos_unpaired!=0 || neg_unpaired!=0;
                this->ref = ref;
                this->pos_paired = pos_paired;
                this->neg_paired = neg_paired;
                this->pos_unpaired = pos_unpaired;
                this->neg_unpaired = neg_unpaired;
                return *this;
            }
#if 0
            Options& margin_terms( 
                        const std::vector<std::pair<u_int32_t, u_int32_t>>& ref2, 
                        float pos_paired=0, float neg_paired=0, float pos_unpaired=0, float neg_unpaired=0)
            {
                this->use_margin = pos_paired!=0 || neg_paired!=0 || pos_unpaired!=0 || neg_unpaired!=0;
                this->ref2 = ref2;
                this->pos_paired = pos_paired;
                this->neg_paired = neg_paired;
                this->pos_unpaired = pos_unpaired;
                this->neg_unpaired = neg_unpaired;
                return *this;
            }
#endif
            Options& score_paired_position(const std::vector<float>& sc)
            {
                if (sc.size() > 0)
                    this->score_paired_position_ = sc;
                return *this;
            }

            Options& set_allowed_pair(char x, char y)
            {
                allowed_pairs_[x][y] = allowed_pairs_[y][x] = true;
                return *this;
            }

            auto additional_paired_score(u_int32_t i, uint32_t j) const
            {
                auto s = 0.0f;
                if (use_margin) 
                    s += ref[i]==j ? -pos_paired : neg_paired;
                if (score_paired_position_.size() > 0)
                    s += score_paired_position_[i-1] + score_paired_position_[j-1];
                return s;
            }

            auto make_constraint(const std::string& seq, bool canonical_only=true)
                -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>> const;
            auto make_additional_scores(size_t L)
                -> std::tuple<TriMatrix<float>, std::vector<std::vector<float>>> const;
            bool allow_paired(char x, char y);
        };

    public:
        static auto parse_paren(const std::string& paren) 
            -> std::vector<u_int32_t>;
        static auto make_paren(const std::vector<u_int32_t>& p) -> std::string;
};