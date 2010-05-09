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
#include "PosteriorSampler.hpp"
#include <Models/ParamTypes.hpp>
namespace BOOM{

  class Sampler;
  class Model;
  class VectorModel;

  class DefaultPosteriorSampler : virtual public PosteriorSampler{
  public:
    DefaultPosteriorSampler(Ptr<Params>, Ptr<Sampler>, Ptr<Model>);
    virtual void draw();
    virtual double logpri()const;
  private:
    Ptr<Params> prm;
    Ptr<Sampler> sam;
    Ptr<Model> pri;     // must supply pri->pdf(prm, true)
    mutable Vec wsp;    // temporary workspace
  };

  class VectorPosteriorSampler : virtual public PosteriorSampler{
  public:
    VectorPosteriorSampler(ParamVec, Ptr<Sampler>, Ptr<VectorModel>);
    virtual void draw();
    virtual double logpri()const;
  private:
    ParamVec prms;
    Ptr<Sampler> sam;
    Ptr<VectorModel> pri;
    mutable Vec wsp;
  };

}
