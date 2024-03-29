/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_ZERO_INFLATED_GAMMA_POSTERIOR_SAMPLER_HPP_
#define BOOM_ZERO_INFLATED_GAMMA_POSTERIOR_SAMPLER_HPP_

#include <Models/ZeroInflatedGammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/BetaBinomialSampler.hpp>
#include <Models/PosteriorSamplers/GammaPosteriorSampler.hpp>

namespace BOOM {

  class ZeroInflatedGammaPosteriorSampler
      : public PosteriorSampler {
   public:
    ZeroInflatedGammaPosteriorSampler(
        ZeroInflatedGammaModel *model,
        Ptr<BetaModel> prior_for_positive_probability,
        Ptr<DoubleModel> prior_for_gamma_mean,
        Ptr<DoubleModel> prior_for_gamma_shape);
    virtual double logpri() const;
    virtual void draw();
   private:
    ZeroInflatedGammaModel *model_;
    Ptr<BetaBinomialSampler> binomial_sampler_;
    Ptr<GammaPosteriorSampler> gamma_sampler_;
  };

}  // namespace BOOM

#endif //  BOOM_ZERO_INFLATED_GAMMA_POSTERIOR_SAMPLER_HPP_
