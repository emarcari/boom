#ifndef BOOM_STATE_SPACE_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_STATE_MODEL_HPP
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

#include <Models/ModelTypes.hpp>
#include <LinAlg/VectorView.hpp>
#include <uint.hpp>

namespace BOOM{
  class StateModel
      : virtual public Model
  {
  public:
    virtual ~StateModel(){}
    virtual StateModel * clone()const=0;
    virtual void observe_state(const ConstVectorView & now, const ConstVectorView & next)=0;
    virtual uint state_size()const=0;
    virtual uint innovation_size()const=0;
  private:
  };
}



#endif// BOOM_STATE_SPACE_STATE_MODEL_HPP
