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

#include <Models/PosteriorSamplers/ZeroInflatedGammaPosteriorSampler.hpp>
#include <distributions.hpp>

namespace BOOM {

  ZeroInflatedGammaPosteriorSampler::ZeroInflatedGammaPosteriorSampler(
      ZeroInflatedGammaModel *model,
      Ptr<BetaModel> prior_for_nonzero_probability,
      Ptr<DoubleModel> prior_for_gamma_mean,
      Ptr<DoubleModel> prior_for_gamma_shape)
      : model_(model),
        binomial_sampler_(new BetaBinomialSampler(
            model->Binomial_model().get(),
            prior_for_nonzero_probability)),
        gamma_sampler_(new GammaPosteriorSampler(
            model->Gamma_model().get(),
            prior_for_gamma_mean,
            prior_for_gamma_shape))
  {}

  double ZeroInflatedGammaPosteriorSampler::logpri() const {
    return binomial_sampler_->logpri() + gamma_sampler_->logpri();
  }

  void ZeroInflatedGammaPosteriorSampler::draw() {
    binomial_sampler_->draw();
    gamma_sampler_->draw();
  }

}
