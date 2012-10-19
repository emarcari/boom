/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_MODELS_POSTERIOR_SAMPLERS_GAMMA_POSTERIOR_SAMPLER_HPP_
#define BOOM_MODELS_POSTERIOR_SAMPLERS_GAMMA_POSTERIOR_SAMPLER_HPP_

#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Samplers/ScalarSliceSampler.hpp>

namespace BOOM {

  // The GammaPosteriorSampler assumes indepdendent priors on a/b and
  // a.

  class GammaPosteriorSampler : public PosteriorSampler {
   public:
    GammaPosteriorSampler(GammaModel *model,
                          Ptr<DoubleModel> mean_prior,
                          Ptr<DoubleModel> alpha_prior);
    virtual void draw();
    virtual double logpri()const;

    double logpost_mean(double mean)const;
    double logpost_alpha(double shape)const;
   private:
    GammaModel *model_;
    Ptr<DoubleModel> mean_prior_;
    Ptr<DoubleModel> alpha_prior_;
    ScalarSliceSampler mean_sampler_;
    ScalarSliceSampler alpha_sampler_;
  };

}  // namespace BOOM

#endif //  BOOM_MODELS_POSTERIOR_SAMPLERS_GAMMA_POSTERIOR_SAMPLER_HPP_
