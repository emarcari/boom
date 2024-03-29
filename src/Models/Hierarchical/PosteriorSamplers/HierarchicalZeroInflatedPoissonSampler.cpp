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

#include <Models/Hierarchical/PosteriorSamplers/HierarchicalZeroInflatedPoissonSampler.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {

typedef HierarchicalZeroInflatedPoissonSampler HZIPS;

  HZIPS::HierarchicalZeroInflatedPoissonSampler(
      HierarchicalZeroInflatedPoissonModel *model,
      Ptr<DoubleModel> lambda_mean_prior,
      Ptr<DoubleModel> lambda_sample_size_prior,
      Ptr<DoubleModel> zero_probability_mean_prior,
      Ptr<DoubleModel> zero_probability_sample_size_prior)
      : model_(model),
        lambda_mean_prior_(lambda_mean_prior),
        lambda_sample_size_prior_(lambda_sample_size_prior),
        zero_probability_mean_prior_(zero_probability_mean_prior),
        zero_probability_sample_size_prior_(zero_probability_sample_size_prior),
        lambda_prior_sampler_(
            model_->prior_for_poisson_mean(),
            lambda_mean_prior_,
            lambda_sample_size_prior_),
        zero_probability_prior_sampler_(
            model_->prior_for_zero_probability(),
            zero_probability_mean_prior_,
            zero_probability_sample_size_prior_)
  {
    Ptr<GammaModel> lambda_prior = model_->prior_for_poisson_mean();
    Ptr<BetaModel> beta_prior = model_->prior_for_zero_probability();
  }

  //----------------------------------------------------------------------
  void HierarchicalZeroInflatedPoissonSampler::draw() {
    GammaModel *lambda_prior = model_->prior_for_poisson_mean();
    lambda_prior->clear_data();

    BetaModel *zero_probability_prior = model_->prior_for_zero_probability();
    zero_probability_prior->clear_data();

    for (int i = 0; i < model_->number_of_groups(); ++i) {
      ZeroInflatedPoissonModel *data_level_model = model_->data_model(i);
      if (data_level_model->number_of_sampling_methods() == 0) {
        NEW(ZeroInflatedPoissonSampler, sampler)(
            data_level_model,
            lambda_prior,
            zero_probability_prior);
        data_level_model->set_method(sampler);
      }
      data_level_model->sample_posterior();
      lambda_prior->suf()->update_raw(data_level_model->lambda());
      zero_probability_prior->suf()->update_raw(
          data_level_model->zero_probability());
    }

    lambda_prior_sampler_.draw();
    zero_probability_prior_sampler_.draw();
  }

  //----------------------------------------------------------------------
  double HierarchicalZeroInflatedPoissonSampler::logpri()const{
    double lambda_mean = model_->poisson_prior_mean();
    double lambda_sample_size = model_->poisson_prior_sample_size();
    double zero_probability_prior_mean = model_->zero_probability_prior_mean();
    double zero_probability_prior_sample_size =
        model_->zero_probability_prior_sample_size();
    if (lambda_mean <= 0
        || lambda_sample_size <= 0
        || zero_probability_prior_mean <= 0
        || zero_probability_prior_mean >= 1
        || zero_probability_prior_sample_size <= 0) {
      return negative_infinity();
    }
    return lambda_mean_prior_->logp(lambda_mean)
        + lambda_sample_size_prior_->logp(lambda_sample_size)
        + zero_probability_mean_prior_->logp(zero_probability_prior_mean)
        + zero_probability_sample_size_prior_->logp(
            zero_probability_prior_sample_size);
  }

} // namespace BOOM
