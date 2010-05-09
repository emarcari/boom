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
#ifndef BOOM_LOCAL_LINEAR_TREND_CONJ_SAMPLER_HPP
#define BOOM_LOCAL_LINEAR_TREND_CONJ_SAMPLER_HPP

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/GammaModel.hpp>

namespace BOOM{

  class LocalLinearTrend;

  class LocalLinearTrendConjSampler
      : public PosteriorSampler
  {
   public:
    LocalLinearTrendConjSampler(LocalLinearTrend *, Ptr<GammaModelBase> );
    virtual void draw();
    virtual double logpri()const;
    virtual void find_posterior_mode();
   private:
    LocalLinearTrend * mod_;
    Ptr<GammaModelBase> pri_; // prior for siginv
  };

}

#endif// BOOM_LOCAL_LINEAR_TREND_CONJ_SAMPLER_HPP
