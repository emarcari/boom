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
#ifndef BOOM_FREQ_DIST_HPP
#define BOOM_FREQ_DIST_HPP

#include <vector>
#include <BOOM.hpp>
#include <cpputil/Ptr.hpp>

namespace BOOM{

  class CategoricalData;

  class FreqDist{
  public:
    FreqDist(const std::vector<Ptr<CategoricalData> > &y);
    FreqDist(const std::vector<uint> &y);
  private:
    std::vector<string> labs_;
    std::vector<uint> counts_;
    friend ostream & operator<<(ostream &,const FreqDist &);
  };

  ostream & operator<<(ostream &, const FreqDist &);
}
#endif// BOOM_FREQ_DIST_HPP
