/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_CPP_MATH_UTILS_H
#define BOOM_CPP_MATH_UTILS_H

#include <LinAlg/Types.hpp>
#include <cmath>

#ifdef _MSC_VER
namespace Rmath{
  double log1p(double x);
}

namespace std{
  inline bool isnan(double x){
    return x!=x;
  }
  inline double log1p(double x){return Rmath::log1p(x);}
}
#endif


namespace BOOM{
  inline int I(int r, int s){ return r==s ? 1:0;}
  double safelog(double x);
  double infinity(int sgn=1);
  bool finite(double x);
  using std::isnan;
}
#endif // BOOM_CPP_MATH_UTILS_H


