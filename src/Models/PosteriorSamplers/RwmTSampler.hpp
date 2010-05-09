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
#ifndef BOOM_RWM_T_SAMPLER_HPP
#define
#include "PosteriorSampler.hpp"
#include <Models/ParamTypes.hpp>

namespace BOOM{
  class LoglikeTF;
  class LogPostTF;
  class SliceSampler;
  class LoglikeModel;
  class VectorModel;



  class PosteriorRwmTSampler : virtual public PosteriorSampler{
  public:
    PosteriorRwmTSampler(Ptr<Params>, Ptr<LoglikeModel>, Ptr<VectorModel>,
			  bool unimodal=false);
    PosteriorRwmTSampler(const ParamVec &,
			  Ptr<LoglikeModel>,
			  Ptr<VectorModel>,
			  bool unimodal=false);

    PosteriorRwmTSampler(Ptr<LoglikeModel>, Ptr<VectorModel>, bool unimodal=false);
    virtual void draw();
    virtual double logpri()const;

    void set_proposal_variance(const Spd &);
    void set_proposal_ivar(const Spd &);
  private:
    ParamVec prms;
    Ptr<LoglikeTF> loglike;
    Ptr<VectorModel> pri;
    Ptr<LogPostTF> logpost;
    mutable Vec x;   // temporary workspace
    Ptr<RWM> sampler;
  };
}
#endif //BOOM_RWM_T_SAMPLER_HPP

