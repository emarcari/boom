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


#ifndef BOOM_STATE_SPACE_HOMOGENEOUS_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_HOMOGENEOUS_STATE_MODEL_HPP

#include <Models/StateSpace/StateModel.hpp>
#include <LinAlg/Types.hpp>
#include <LinAlg/SpdMatrix.hpp>

namespace BOOM{

  class HomogeneousStateModel
    : public StateModel
  {

  public:
    HomogeneousStateModel(uint state_size, uint innovation_size);
    HomogeneousStateModel * clone()const=0;

    const Vec & Z()const;
    const Mat & T()const;
    const Mat & R()const;
    const Spd & Q()const;

    void set_Z(const Vec & );
    void set_T(const Mat &);
    void set_R(const Mat &);
    void set_Q(const Spd &);

   private:
    Vec Z_;    // picks off this state model's contribution to y[t]
    Mat T_;    // mean transition function
    Mat R_;    // State variance is R Q R^T
    Spd Q_;    // state innovation variance
  };
}

#endif// BOOM_STATE_SPACE_HOMOGENEOUS_STATE_MODEL_HPP
