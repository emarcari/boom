/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include <distributions/rng.hpp>

namespace BOOM{

class binomial_distribution{
 public:
  binomial_distribution(uint n, double p);
  uint operator()(RNG &);

 private:
  double psave;
  uint nsave;

  double c, fm, npq, p1, p2, p3, p4, qn;
  double xl, xll, xlr, xm, xr;
  int m;

  uint np_large(RNG &);
  uint np_small(RNG &);
  uint n_small(RNG &);
};

}
