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

#ifndef BOOM_MOMENTS_HPP
#define BOOM_MOMENTS_HPP

#include <BOOM.hpp>
#include <vector>
#include <LinAlg/Types.hpp>

namespace BOOM{

  Vec mean(const Mat &m);
  Spd var(const Mat &m);
  Spd cor(const Mat &m);

  double mean(const Vec &x);
  double var(const Vec &x);

  double mean(const std::vector<double> &x);
  double var(const std::vector<double> &x);

}
#endif // BOOM_MOMENTS_HPP
