#include "bpseq.h"
#include <iostream>
#include <fstream>
#include <stack>
#include <cstring>
#include <cerrno>

BPSEQ::
BPSEQ()
  : seq_(), bp_()
{
}

BPSEQ::
BPSEQ(const std::string& seq, const std::string& paren)
  : seq_(seq), bp_(seq.size()+1,0)
{
  std::stack<unsigned int> st;
  for (unsigned int i=0; i!=paren.size(); ++i)
  {
    switch (paren[i])
    {
      case '(':
        st.push(i); break;
      case ')':
      {
        int j=st.top();
        st.pop();
        bp_[i+1]=j+1;
        bp_[j+1]=i+1;
      }
      break;
      default: break;
    }
  }
}

bool
BPSEQ::
load(const char* fname)
{
  int i, j, l=0;
  char c;
  std::ifstream is(fname);
  if (!is) throw std::runtime_error(std::string(strerror(errno)) + ": " + std::string(fname));
  while (is >> i >> c >> j) l=std::max(l, i);
  seq_.clear(); seq_.resize(l);
  bp_.clear(); bp_.resize(l+1,0);
  is.close();
  
  is.open(fname);
  if (!is.is_open()) return false;
  while (is >> i >> c >> j)
  {
    seq_[i-1] = c;
    bp_[i] = j;
  }

  return true;
}

bool
BPSEQ::
save(const char* fname) const
{
  std::ofstream os(fname);
  if (!os.is_open()) return false;
  for (unsigned int i=1; i!=bp_.size(); ++i)
    os << i << " " << seq_[i-1] << " " << bp_[i] << std::endl;
  return true;
}

std::vector<BPSEQ>
BPSEQ::
load_from_list(const char* fname)
{
  std::string f;
  std::ifstream is(fname);
  if (!is) throw std::runtime_error(std::string(strerror(errno)) + ": " + std::string(fname));
  std::vector<BPSEQ> ret;
  while (is >> f)
  {
    ret.emplace_back();
    ret.back().load(f.c_str());
  }
  return ret;
}

#ifdef TEST
int
main(int argc, char* argv[])
{
  BPSEQ bpseq("GGCAUUCC",
              "((....))");
  bpseq.save(argv[1]);
  return 0;
}
#endif
