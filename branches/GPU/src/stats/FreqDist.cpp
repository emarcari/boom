/*
  Copyright (C) 2007 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/
#include "FreqDist.hpp"
#include <Models/CategoricalData.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace BOOM{

  inline string u2str(uint u){
    std::ostringstream out;
    out << u;
    return out.str();
  }

  typedef FreqDist FD;
  FD::FreqDist(const std::vector<Ptr<CategoricalData> > &y)
    : labs_(y[0]->labels()),
      counts_(y[0]->nlevels())
  {
    for(uint i=0; i<y.size(); ++i){
      ++counts_[y[i]->value()];
    }
  }

  FD::FreqDist(const std::vector<uint> &y){
    std::vector<uint> x(y);
    std::sort(x.begin(), x.end());
    uint last = x[0];
    string lab = u2str(last);
    uint count = 1;
    for(uint i=1; i<x.size(); ++i){
      if(x[i]!=last){
	counts_.push_back(count);
	labs_.push_back(lab);
	count = 1;
	last = x[i];
	lab = u2str(last);
      }else{
	++count;
      }
    }
    counts_.push_back(count);
    labs_.push_back(lab);
  }


  ostream & operator<<(ostream & out, const FD &d){
    uint N = d.labs_.size();
    uint labfw=0;
    uint countfw=0;
    for(uint i = 0; i<N; ++i){
      uint len = d.labs_[i].size();
      if(len > labfw) labfw = len;

      string s = u2str(d.counts_[i]);
      len = s.size();
      if(len > countfw) countfw = len;
    }
    labfw += 2;
    countfw +=2;

    for(uint i=0; i<N; ++i){
      out << std::setw(labfw) << d.labs_[i]
	  << std::setw(countfw) << d.counts_[i]
	  << endl;
    }
    return out;
  }
}
