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

#include <Models/StateSpace/HomogeneousStateModel.hpp>

namespace BOOM{

  typedef HomogeneousStateModel HSM;
  HSM::HomogeneousStateModel(uint state, uint innovation)
      : Z_(state),
        T_(state,state),
        R_(state, innovation),
        Q_(innovation)
  {}

  const Vec & HSM::Z()const{return Z_;}
  const Mat & HSM::T()const{return T_;}
  const Mat & HSM::R()const{return R_;}
  const Spd & HSM::Q()const{return Q_;}

  void HSM::set_Z(const Vec &Z){Z_ = Z;}
  void HSM::set_T(const Mat &T){T_ = T;}
  void HSM::set_R(const Mat &R){R_ = R;}
  void HSM::set_Q(const Spd &Q){Q_ = Q;}
}
