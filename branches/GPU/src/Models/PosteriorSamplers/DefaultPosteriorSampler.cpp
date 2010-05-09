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
#include "DefaultPosteriorSampler.hpp"
#include <Models/VectorModel.hpp>

#include <Samplers/Sampler.hpp>

namespace BOOM{
  typedef DefaultPosteriorSampler DPS;

  DPS::DefaultPosteriorSampler(Ptr<Params> p, Ptr<Sampler> s, Ptr<Model> m)
    : prm(p),
      sam(s),
      pri(m),
      wsp(p->vectorize())
  {}

  void DPS::draw(){
    wsp = prm->vectorize();
    wsp = sam->draw(wsp);
    prm->unvectorize(wsp);
  }

  double DPS::logpri()const{ return pri->pdf(prm, true); }

  typedef VectorPosteriorSampler VPS;

  VPS::VectorPosteriorSampler(ParamVec p, Ptr<Sampler> s, Ptr<VectorModel> m)
    : prms(p),
      sam(s),
      pri(m),
      wsp(vectorize(p))
  {}

  void VPS::draw(){
    wsp = vectorize(prms);
    wsp = sam->draw(wsp);
    unvectorize(prms, wsp);
  }

  double VPS::logpri()const{
    wsp = vectorize(prms);
    return pri->logp(wsp);}

}
