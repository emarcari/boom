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
#ifndef BOOM_WEIGHTED_RESIDUAL_VARIANCE_STATE_SPACE_MODEL_HPP
#define BOOM_WEIGHTED_RESIDUAL_VARIANCE_STATE_SPACE_MODEL_HPP

#include "StateSpaceData.hpp"
#include "StateSpaceModel.hpp"

namespace BOOM{

  class WeightedStateSpaceModel
    : virtual public StateSpaceModel{
    // mix-in class for previously defined state space models
    // residual variance $H_t = Sigma/w$
  public:
    WeightedStateSpaceModel * clone()const=0;

    virtual Spd Sigma()const=0;
    virtual Spd residual_variance(Ptr<StateSpaceData> dp)const;
  };
}
#endif //BOOM_WEIGHTED_RESIDUAL_VARIANCE_STATE_SPACE_MODEL_HPP
