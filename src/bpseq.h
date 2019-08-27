// BPSEQ
#pragma once

#include <string>
#include <vector>

class BPSEQ
{
public:
  BPSEQ();
  BPSEQ(const std::string& seq, const std::string& paren);

  bool load(const char* fname);
  bool save(const char* fname) const;

  const std::string& seq() const { return seq_; }
  const auto& bp() const { return bp_; }
  std::string stru(char unpaired='.', const char* paired="()") const;

  static std::vector<BPSEQ> load_from_list(const char* fname);

private:
  std::string seq_;
  std::vector<u_int32_t> bp_;
};

