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
#ifndef BOOM_REFRESHER_HPP
#define BOOM_REFRESHER_HPP

#include <cpputil/Ptr.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>

namespace BOOM{
  template <class DAT, class SUF>
  class Refresher : public PosteriorSampler{
    // this 'sampler' is used for models that need to refresh their
    // sufficent statistics prior to drawing.
  public:
    Refresher(Ptr<SufstatDataPolicy<DAT,SUF> > Mod)
      : mod(Mod)
    {}

    void draw(){mod->refresh_suf();}
    double logpri()const{return 0;}
  private:
    Ptr<SufstatDataPolicy<DAT,SUF> > mod;
  };
}
#endif// BOOM_REFRESHER_HPP
