/*
 * Copyright (C) 2008-2019 Kengo Sato
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif
#include "fasta.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <cstring>
#include <cassert>

typedef unsigned int uint;

//static
std::vector<Fasta>
Fasta::
load(const char* file)
{
  std::vector<Fasta> data;
  std::string line, name, seq, str;
  std::ifstream ifs(file);
  while (std::getline(ifs, line)) {
    if (line[0]=='>') {         // header
      if (!name.empty()) {
        assert(str.size()==0 || seq.size()==str.size());
#if 0
        std::cout << "name: " << name << std::endl
                  << " seq: " << seq << std::endl
                  << " str: " << str << std::endl;
#endif
        data.emplace_back(name, seq, str);
      }

      name=line.substr(1);
      seq.clear();
      str.clear();
      continue;
    } 

    //if (std::strchr("()[].?xle ", line[0])==NULL) { // seq
    if (std::strchr("()[]<>|x.", line[0])==NULL) { // seq
      uint i;
      for (i=0; i!=line.size(); ++i)
        if (!isalpha(line[i])) break;
      seq+=line.substr(0, i);
    } else {
      uint i;
      for (i=0; i!=line.size(); ++i)
        //if (std::strchr("()[].?xle ", line[i])==NULL) break;
        if (std::strchr("()[]<>|x.", line[i])==NULL) break;
      str+=line.substr(0, i);
    }
  }
  
  if (!name.empty()) {
#if 0
    std::cout << "name: " << name << std::endl
              << " seq: " << seq << std::endl
              << " str: " << str << std::endl;
#endif
    data.emplace_back(name, seq, str);
  }

  return data;
}

