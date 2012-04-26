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

#include <Models/PosteriorSamplers/BetaBinomialPosteriorSampler.hpp>
#include <boost/bind.hpp>

namespace BOOM{

  typedef BetaBinomialPosteriorSampler BBPS;

  BBPS::BetaBinomialPosteriorSampler(
      BetaBinomialModel *model,
      Ptr<BetaModel> probability_prior_distribution,
      Ptr<DoubleModel> sample_size_prior_distribution)
      : model_(model),
        probability_prior_distribution_(probability_prior_distribution),
        sample_size_prior_distribution_(sample_size_prior_distribution),
        probability_sampler_(boost::bind(
            &BetaBinomialPosteriorSampler::logp_prob, this, _1)),
        sample_size_sampler_(boost::bind(
            &BetaBinomialPosteriorSampler::logp_sample_size, this, _1))
  {
    probability_sampler_.set_limits(0,1);
    sample_size_sampler_.set_lower_limit(0);
  }

  double BBPS::logpri()const{
    double prob = model_->prior_mean();
    double sample_size = model_->prior_sample_size();
    return probability_prior_distribution_->logp(prob) +
        sample_size_prior_distribution_->logp(sample_size);
  }

  void BBPS::draw(){
    double prob = model_->prior_mean();
    prob = probability_sampler_.draw(prob);
    model_->set_prior_mean(prob);

    double sample_size = model_->prior_sample_size();
    sample_size  = sample_size_sampler_.draw(sample_size);
    model_->set_prior_sample_size(sample_size);
  }

  double BBPS::logp_sample_size(double sample_size)const{
    double prob = model_->prior_mean();
    double a = prob * sample_size;
    double b = sample_size - a;
    return  probability_prior_distribution_->logp(prob)
        + sample_size_prior_distribution_->logp(sample_size)
        + model_->loglike(a, b);
  }

  double BBPS::logp_prob(double prob)const{
    double sample_size = model_->prior_sample_size();
    double a = prob * sample_size;
    double b = sample_size - a;
    return  probability_prior_distribution_->logp(prob)
        + sample_size_prior_distribution_->logp(sample_size)
        + model_->loglike(a, b);
  }

}
