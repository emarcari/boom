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

#include <Models/PosteriorSamplers/GammaPosteriorSampler.hpp>
#include <boost/bind.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

GammaPosteriorSampler::GammaPosteriorSampler(GammaModel *model,
                                             Ptr<DoubleModel> mean_prior,
                                             Ptr<DoubleModel> alpha_prior)
    : model_(model),
      mean_prior_(mean_prior),
      alpha_prior_(alpha_prior),
      mean_sampler_(boost::bind(&GammaPosteriorSampler::logpost_mean,
                                this,
                                _1),
                    true),
      alpha_sampler_(boost::bind(&GammaPosteriorSampler::logpost_alpha,
                                 this,
                                 _1),
                     true)
{
  mean_sampler_.set_lower_limit(0);
  alpha_sampler_.set_lower_limit(0);
}

  double GammaPosteriorSampler::logpost_mean(double mean)const{
    if (mean < 0) return infinity(-1);
    double a = model_->alpha();
    if (mean <= 0 &&  a > 0) return infinity(-1);
    double ans = mean_prior_->logp(mean);
    double b = a / mean;
    ans += model_->loglikelihood(a, b);
    return ans;
  }

  double GammaPosteriorSampler::logpost_alpha(double alpha)const{
    if (alpha < 0) return infinity(-1);
    double ans = alpha_prior_->logp(alpha);
    if (ans <= infinity(-1)) return ans;
    ans += model_->loglikelihood(alpha, model_->beta());
    return ans;
  }

  void GammaPosteriorSampler::draw(){
    double alpha = alpha_sampler_.draw(model_->alpha());
    model_->set_alpha(alpha);

    double mean = alpha / model_->beta();
    mean = mean_sampler_.draw(mean);
    double beta = alpha / mean;
    model_->set_beta(beta);
  }

  double GammaPosteriorSampler::logpri()const{
    double a = model_->alpha();
    double mean = a / model_->beta();
    return mean_prior_->logp(mean) + alpha_prior_->logp(a);
  }

}  // namespace BOOM
