/*
  Copyright (C) 2006 Steven L. Scott

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

#include "UniformCorrelationModel.hpp"
#include <cpputil/math_utils.hpp>
#include <distributions.hpp>

//using namespace boost::lambda;
namespace BOOM{
  typedef UniformCorrelationModel UCM;

  UCM::UniformCorrelationModel(uint dim)
    : dim_(dim)
  {}

  UCM::UniformCorrelationModel(const UCM &rhs)
    : Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      CorrModel(rhs),
      dim_(rhs.dim_)
  {}

  UCM * UCM::clone()const{return new UCM(*this);}
  void UCM::initialize_params(){}

  double UCM::pdf(const Corr &m, bool logscale)const{
    double ans = m.is_pos_def() ? 0.0 : BOOM::infinity(-1);
    return logscale ? ans : exp(ans);
  }

  double UCM::pdf(Ptr<Data> dp, bool logscale)const{
    return pdf(DAT(dp), logscale);}

  uint UCM::dim()const{return dim_;}

  Corr UCM::sim()const{
    return LinAlg::random_cor(dim());
  }
}
