/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include <Models/BetaBinomialModel.hpp>
#include <Models/BetaModel.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Samplers/ScalarSliceSampler.hpp>

namespace BOOM{

  // This is a posterior sampler for the BetaBinomialModel.  It
  // differs from the BetaBinomialSampler, which is a sampler for the
  // binomial model based on a beta prior.
  class BetaBinomialPosteriorSampler
      : public PosteriorSampler {
   public:
    BetaBinomialPosteriorSampler(
        BetaBinomialModel *model,
        Ptr<BetaModel> probability_prior_distribution,
        Ptr<DoubleModel> sample_size_prior_distribution);

    virtual void draw();
    virtual double logpri()const;

    // Full conditional distributions of the probability and sample
    // size parameters.
    double logp_prob(double prob)const;
    double logp_sample_size(double sample_size)const;
   private:
    BetaBinomialModel *model_;
    Ptr<BetaModel> probability_prior_distribution_;
    Ptr<DoubleModel> sample_size_prior_distribution_;

    ScalarSliceSampler probability_sampler_;
    ScalarSliceSampler sample_size_sampler_;
  };


}
