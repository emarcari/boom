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

#ifndef BOOM_POISSON_GAMMA_POSTERIOR_SAMPLER_HPP_
#define BOOM_POISSON_GAMMA_POSTERIOR_SAMPLER_HPP_

#include <Models/DoubleModel.hpp>
#include <Models/PoissonGammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Samplers/ScalarSliceSampler.hpp>

namespace BOOM {

  class PoissonGammaPosteriorSampler : public PosteriorSampler {
   public:
    PoissonGammaPosteriorSampler(PoissonGammaModel *model,
                                 Ptr<DoubleModel> mean_prior_distribution,
                                 Ptr<DoubleModel> sample_size_prior);

    virtual void draw();
    virtual double logpri()const;

    double logp(double prior_mean, double prior_sample_size)const;

   private:
    PoissonGammaModel *model_;

    // Prior for a/b
    Ptr<DoubleModel> prior_mean_prior_distribution_;

    // Prior for b
    Ptr<DoubleModel> prior_sample_size_prior_distribution_;

    ScalarSliceSampler prior_mean_sampler_;
    ScalarSliceSampler prior_sample_size_sampler_;
  };

}  // namespace BOOM
#endif  //  BOOM_POISSON_GAMMA_POSTERIOR_SAMPLER_HPP_
