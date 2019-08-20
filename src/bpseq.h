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
  const std::vector<int>& bp() const { return bp_; }
  std::string stru() const;

  static std::vector<BPSEQ> load_from_list(const char* fname);

private:
  std::string seq_;
  std::vector<int> bp_;
};

