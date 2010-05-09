/*
  Copyright (C) 2008 Steven L. Scott

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

#ifndef BOOM_KALMAN_TOOLS_HPP
#define BOOM_KALMAN_TOOLS_HPP

#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>


namespace BOOM{

  double scalar_kalman_update(double y, Vec &a, Spd &P, Vec &K, double &F, double &v, bool missing,
                              const Vec & Z, double H, const Mat & T, Mat & L, const Spd & RQR);


  void scalar_kalman_smoother_update(Vec &a, Spd &P, const Vec & K, double F, double v,
                                     const Vec & Z, const Mat &T,
                                     Vec & r, Mat & N, Mat & L);

}
#endif// BOOM_KALMAN_TOOLS_HPP
