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

#include "ScalarLogpostTF.hpp"
#include <TargetFun/Loglike.hpp>
#include <Models/DoubleModel.hpp>

namespace BOOM{
  typedef ScalarLogpostTF SLT;

  SLT::ScalarLogpostTF(LoglikeModel * m, Ptr<DoubleModel> Pri)
    : loglike(LoglikeTF(m)),
      pri(Pri)
  { }

  double SLT::operator()(const Vec &x)const{
    double ans = loglike(x);
    ans += pri->logp(x[0]);
    return ans;
  }

  double SLT::operator()(double x)const{
    Vec v(1,x);
    double ans = loglike(v);
    ans += pri->logp(x);
    return ans;
  }

}
