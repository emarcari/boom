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

#include <Models/Bart/PosteriorSamplers/LogitBartPosteriorSampler.hpp>
#include <distributions.hpp>
#include <cpputil/report_error.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {
  namespace Bart {

    LogitResidualData::LogitResidualData(Ptr<BinomialRegressionData> dp,
                                         double original_prediction)
        : ResidualRegressionData(dp->Xptr().get()),
          original_data_(dp.get()),
          sum_of_information_(0.0),
          information_weighted_sum_(0.0),
          prediction_(original_prediction)
    {}

    void LogitResidualData::add_to_residual(double value) {
      prediction_ -= value;
    }

    void LogitResidualData::add_to_logit_suf(
        LogitSufficientStatistics &suf) const {
      suf.update(*this);
    }

    void LogitResidualData::set_latent_data(
        double information_weighted_sum_of_latent_logits,
        double sum_of_information) {
      sum_of_information_ = sum_of_information;
      information_weighted_sum_ = information_weighted_sum_of_latent_logits;
    }

    //======================================================================
    LogitSufficientStatistics * LogitSufficientStatistics::clone() const {
      return new LogitSufficientStatistics(*this);
    }

    void LogitSufficientStatistics::clear() {
      sum_of_information_ = 0;
      information_weighted_sum_ = 0;
      information_weighted_prediction_ = 0;
     }

    void LogitSufficientStatistics::update(const ResidualRegressionData &data) {
      data.add_to_logit_suf(*this);
    }

    void LogitSufficientStatistics::update(const LogitResidualData &data) {
      double info = data.sum_of_information();
      sum_of_information_ += info;
      information_weighted_prediction_ += data.prediction() * info;
      information_weighted_sum_ += data.information_weighted_sum();
    }

    double LogitSufficientStatistics::sum_of_information() const {
      return sum_of_information_;
    }

    double LogitSufficientStatistics::information_weighted_sum() const {
      return information_weighted_sum_;
    }

    double LogitSufficientStatistics::information_weighted_residual_sum()const{
      return information_weighted_sum_ - information_weighted_prediction_;
    }

  }  // namespace Bart

  //======================================================================
  LogitBartPosteriorSampler::LogitBartPosteriorSampler(
      LogitBartModel *model,
      double prior_mean_guess,
      double prior_mean_sd,
      double prior_tree_depth_alpha,
      double prior_tree_depth_beta)
      : BartPosteriorSamplerBase(
            model,
            prior_mean_guess,
            prior_mean_sd,
            prior_tree_depth_alpha,
            prior_tree_depth_beta),
        model_(model),
        data_imputer_(new BinomialLogitCltDataImputer(10))
  {}

  //----------------------------------------------------------------------
  void LogitBartPosteriorSampler::draw() {
    check_residuals();
    // The idea here is that the full data is imputed based on all the
    // trees and their current node values.
    //
    // The residuals are going to have to be recalculated each time
    // the data are imputed.
    //
    // The appropriate complete data likelihood is that
    // y[i]=N(eta,sigsq[i]).  The update to each leaf uses
    // (y[i]-eta[i,-])~N(leaf[i], sigsq[i]), so subtracting eta is a
    // must.  The subtraction should be done when leaf[i] is drawn.
    impute_latent_data();
    BartPosteriorSamplerBase::draw();
  }

  //----------------------------------------------------------------------
  // Drawing the mean at a leaf node is like drawing the intercept in
  // a logistic regression model after the other linear effects have
  // been subtracted out.
  double LogitBartPosteriorSampler::draw_mean(Bart::TreeNode *leaf) {
    const Bart::LogitSufficientStatistics * suf(
        dynamic_cast<const Bart::LogitSufficientStatistics *>(
            leaf->compute_suf()));
    double prior_mean = node_mean_prior()->mu();
    double prior_variance = node_mean_prior()->sigsq();
    double ivar = (1.0 / prior_variance) + suf->sum_of_information();
    double posterior_mean =
        (suf->information_weighted_residual_sum()
         + (prior_mean / prior_variance)) / ivar;
    double posterior_sd = sqrt(1.0 / ivar);
    return rnorm_mt(rng(), posterior_mean, posterior_sd);
  }

  //----------------------------------------------------------------------
  double LogitBartPosteriorSampler::log_integrated_likelihood(
      const Bart::SufficientStatisticsBase *suf) const {
    return log_integrated_logit_likelihood(
        dynamic_cast<const Bart::LogitSufficientStatistics *>(
            suf));
  }

  //----------------------------------------------------------------------
  // The following is a derivation of the integrated complete data
  // logit likelihood.
/*
\documentclass{article}
\usepackage{amsmath}
\begin{document}

The logit integrated likelihood assumes $y_i \sim N(\theta,
\sigma^2_{z_i})$, where $z_i$ is the mixture indicator from the
discrete normal approximation to the logit.  If there are $K$
components in the normal mixture approximation, then let $n_k$ denote
the number from mixture component $k$.  Let $w_i = 1 /
\sigma^2_{z_i}$, $w_+ = \sum_i w_i$, $\bar y_w = \sum_i w_i y_i /
w_+$, and $S_w = \sum_i w_i (y_i - \bar y_w)^2$.  Let $v^{-1} =
\frac{1}{\tau^2} + w_+$ denote the posterior variance and $\tilde \mu
= v(w_+\bar y_w + \mu_0/\tau^2)$ the posterior mean.

\begin{equation*}
  \begin{split}
    p(y) & =
    (2\pi)^{-\frac{n+1}{2}}
    \frac{1}{\tau}
    \prod_i w_i^{\frac{1}{2}}
    \int
    \exp \left(
      -\frac{1}{2} \left[
        \frac{(\theta - \mu_0)^2}{\tau^2}
         + \sum_i w_i(y_i - \theta)^2
       \right]
    \right)
    \ d \theta
    \\
    &=    (2\pi)^{-\frac{n+1}{2}}
    \left(\frac{1}{\tau^2}\right)^{\frac{1}{2}}
    \prod_i w_i^{\frac{1}{2}}
    \left(\frac{v}{v}\right)^{\frac{1}{2}}
    \\
    & \qquad
    \times
    \int
    \exp \left(
      -\frac{1}{2}
      \left[
        \theta^2\left(\frac{1}{\tau^2} + \sum_i w_i\right)
        -2\theta\left( \frac{\mu_0}{\tau^2} + \sum_i w_iy_i\right)
        + \frac{\tilde\mu^2}{v}
        \right.
        \right. \\
        & \qquad \qquad \qquad \qquad
        \left.
          \left.
        - \frac{\tilde\mu^2}{v}
         + \sum_i w_iy_i^2
         + \frac{\mu_0^2}{\tau^2}
      \right]
    \right)
    \ d \theta \\
    &=
    (2\pi)^{-\frac{n}{2}}
    \left(\frac{v}{\tau^2}\right)^{\frac{1}{2}}
    \prod_i w_i^{\frac{1}{2}}
    \exp\left(
      -\frac{1}{2}\left[
        \sum_i w_i y_i^2 + \frac{\mu_0^2}{\tau^2} - \frac{\tilde\mu^2}{\tau^2}a
        \right]
      \right)
  \end{split}
\end{equation*}

Thus the log integrated likelihood is

\begin{equation*}
\log p(y) = \frac{1}{2}\left(
  n \log 2\pi + \log v - \log \tau^2 + \sum_i \log w_i
  - \sum_iw_iy_i^2 - \frac{\mu_0^2}{\tau^2} + \frac{\tilde\mu^2}{v}
\right).
\end{equation*}

For the purposes of BART portions that only depend on the data $w_i$
and $y_i$ can be ignored because they cancel out in the split/combine
moves, and thus a minimal log integrated likelihood is

\begin{equation*}
\ell = \frac{1}{2}\left(
  \log v - \log \tau^2 - \frac{\mu_0^2}{\tau^2} + \frac{\tilde\mu^2}{v}
\right).
\end{equation*}

\end{document}
*/
  double LogitBartPosteriorSampler::log_integrated_logit_likelihood(
      const Bart::LogitSufficientStatistics *suf) const {
    double information = suf->sum_of_information();
    if (information <= 0) {
      // The node is empty.  Note that information can never be
      // negative.  The <= comparison is used because it is not safe
      // to ask information == 0 on a double.
      return 0;
    }
    double prior_variance = node_mean_prior()->sigsq();
    double prior_mean = node_mean_prior()->mu();
    double ivar = information + (1.0 / prior_variance);
    double posterior_variance = 1.0 / ivar;
    double posterior_mean =
        (suf->information_weighted_sum() + prior_mean / prior_variance) / ivar;

    double ans =  .5 * (
        log(posterior_variance / prior_variance)
        - (square(prior_mean) / prior_variance)
        + (square(posterior_mean) / posterior_variance));
    return ans;
  }

  //----------------------------------------------------------------------
  void LogitBartPosteriorSampler::clear_residuals() {
    residuals_.clear();
  }

  //----------------------------------------------------------------------
  int LogitBartPosteriorSampler::residual_size() const {
    return residuals_.size();
  }

  //----------------------------------------------------------------------
  Bart::LogitResidualData *
  LogitBartPosteriorSampler::create_and_store_residual(int i) {
    Ptr<BinomialRegressionData> data_point(model_->dat()[i]);
    double original_prediction = model_->predict(data_point->x());
    boost::shared_ptr<Bart::LogitResidualData> residual(
        new Bart::LogitResidualData(data_point, original_prediction));
    residuals_.push_back(residual);
    return residual.get();
  }

  //----------------------------------------------------------------------
  Bart::LogitSufficientStatistics *
  LogitBartPosteriorSampler::create_suf() const {
    return new Bart::LogitSufficientStatistics;
  }

  //----------------------------------------------------------------------
  void LogitBartPosteriorSampler::impute_latent_data() {
    for (int i = 0; i < residuals_.size(); ++i) {
      impute_latent_data_point(residuals_[i].get());
    }
  }

  //----------------------------------------------------------------------
  void LogitBartPosteriorSampler::impute_latent_data_point(DataType *data) {
    double eta = data->prediction();

#ifndef NDEBUG
    double eta_check = model_->predict(data->x());
    if (fabs(eta - eta_check) > .001) {
      report_error("eta and eta_check are far apart.");
    }
#endif

    std::pair<double, double> latent_data =
        data_imputer_->impute(rng(), data->n(), data->y(), eta);
    double information_weighted_sum_of_latent_logits = latent_data.first;
    double information = latent_data.second;
    data->set_latent_data(information_weighted_sum_of_latent_logits,
                          information);
  }

}  // namespace BOOM
