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

#include <Models/MvnModel.hpp>
#include <Models/DataTypes.hpp>
#include <Models/TimeSeries/TimeSeries.hpp>

namespace BOOM{

class NoStorageScalarKalmanFilter{
 public:
  NoStorageScalarKalmanFilter(const Vec &Z, double H, const Mat &T, const Mat &R, const Spd & Q,
                              Ptr<MvnModel> init);

  double logp(const Vec & ts)const;
  double logp(const TimeSeries<DoubleData> & ts)const;
 private:
  const Vec & Z_;
  double H_;
  const Mat & T_;
  Spd RQR_;

  mutable Mat L_;
  mutable Vec a_;
  mutable Spd P_;
  mutable Vec K_;
  mutable double v;
  mutable double F;


  Ptr<MvnModel> init_; // initial_state_distribution
  double update(double y)const;
  void initialize()const;

};
}
