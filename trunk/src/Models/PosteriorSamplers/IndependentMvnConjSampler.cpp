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

#include <Models/PosteriorSamplers/IndependentMvnConjSampler.hpp>
#include <distributions.hpp>

namespace BOOM {

namespace {
  inline double square(double x) {return x * x;}
}

  IndependentMvnConjSampler::IndependentMvnConjSampler(
      IndependentMvnModel *model,
      const Vec &mean_guess,
      const Vec & mean_sample_size,
      const Vec &sd_guess,
      const Vec &sd_sample_size)
      : model_(model),
        mean_prior_guess_(mean_guess),
        mean_prior_sample_size_(mean_sample_size),
        prior_ss_(sd_guess * sd_guess * sd_sample_size),
        prior_df_(sd_sample_size)
  {}

  IndependentMvnConjSampler::IndependentMvnConjSampler(
      IndependentMvnModel *model,
      double mean_guess,
      double mean_sample_size,
      double sd_guess,
      double sd_sample_size)
      : model_(model),
        mean_prior_guess_(model->dim(), mean_guess),
        mean_prior_sample_size_(model->dim(), mean_sample_size),
        prior_ss_(model->dim(), sd_guess * sd_guess * sd_sample_size),
        prior_df_(model->dim(), sd_sample_size)
  {}

  double IndependentMvnConjSampler::logpri()const{
    int dim = model_->dim();
    const Vec &mu(model_->mu());
    const Vec &sigsq(model_->sigsq());
    double ans = 0;
    for(int i = 0; i < dim; ++i){
      ans += dgamma(1.0/sigsq[i], prior_df_[i] / 2, prior_ss_[i] / 2, true);
      ans += dnorm(mu[i],
                   mean_prior_guess_[i],
                   sqrt(sigsq[i] / mean_prior_sample_size_[i]),
                   true);
    }
    return ans;
  }

  void IndependentMvnConjSampler::draw(){
    int dim = model_->dim();
    const IndependentMvnSuf &suf(*(model_->suf()));
    Vec mu(dim);
    Vec sigsq(dim);
    for(int i = 0; i < dim; ++i){
      double n = suf.n(i);
      double ybar = suf.ybar(i);
      double v = suf.sample_var(i);

      double kappa = mean_prior_sample_size_[i];
      double mu0 = mean_prior_guess_[i];
      double df = prior_df_[i];
      double ss = prior_ss_[i];

      df += n;
      double mu_hat = (n * ybar + kappa * mu0) / (n + kappa);
      ss += (n-1)*v  +  n * kappa * pow(ybar - mu0, 2) / (n + kappa);

      sigsq[i] = 1.0/rgamma_mt(rng(), df/2, ss/2);
      v = sigsq[i] / (n+kappa);
      mu[i] = rnorm_mt(rng(), mu_hat, sqrt(v));
    }
    model_->set_mu(mu);
    model_->set_sigsq(sigsq);
  }
}
