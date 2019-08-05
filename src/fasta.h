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

#pragma once

#include <string>
#include <vector>

class Fasta
{
public:
  Fasta() : name_(), seq_(), str_() { }

  Fasta(const std::string& name,
	const std::string& seq,
	const std::string& str="")
    : name_(name), seq_(seq), str_(str)
  { }

  Fasta(const Fasta& fa)
    : name_(fa.name_), seq_(fa.seq_), str_(fa.str_)
  { }
    
  Fasta&
  operator=(const Fasta& fa)
  {
    if (this != &fa) {
      name_ = fa.name_;
      seq_ = fa.seq_;
      str_ = fa.str_;
    }
    return *this;
  }

  const std::string& name() const { return name_; }
  const std::string& seq() const { return seq_; }
  std::string& seq() { return seq_; }
  const std::string& str() const { return str_; }
  unsigned int size() const { return seq_.size(); }

  static
  std::vector<Fasta> load(const char* file);

  static
  std::vector<Fasta> load(const std::string& file) { return load(file.c_str()); };

private:
  std::string name_;
  std::string seq_;
  std::string str_;
};

