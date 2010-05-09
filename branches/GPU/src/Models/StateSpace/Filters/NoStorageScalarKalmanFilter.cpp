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

#include <Models/StateSpace/Filters/NoStorageScalarKalmanFilter.hpp>
#include <Models/StateSpace/Filters/KalmanTools.hpp>

namespace BOOM{

typedef NoStorageScalarKalmanFilter NSSKF;

NSSKF::NoStorageScalarKalmanFilter(const Vec &Z, double H, const Mat &T, const Mat &R, const Spd & Q,
                                   Ptr<MvnModel> init)
    : Z_(Z),
      H_(H),
      T_(T),
      RQR_(sandwich(R,Q)),
      L_(Z.size(), Z.size()),
      a_(Z.size()),
      P_(Z.size()),
      K_(Z.size()),
      init_(init)
{}


double NSSKF::logp(const Vec & ts)const{
  double ans = 0;
  initialize();
  for(uint i=0; i<ts.size(); ++i) ans += update(ts[i]);
  return ans;
}

double NSSKF::logp(const TimeSeries<DoubleData> & ts)const{
  initialize();
  double ans = 0;
  for(uint i=0; i<ts.size(); ++i) ans += update(ts[i]->value());
  return ans;
}


void NSSKF::initialize()const{
  a_ = init_->mu();
  P_ = init_->Sigma();
}

double NSSKF::update(double y)const{
  double F_, v_;
  return scalar_kalman_update(y,a_,P_,K_,F_, v_, false, Z_, H_, T_, L_, RQR_);
}

}
