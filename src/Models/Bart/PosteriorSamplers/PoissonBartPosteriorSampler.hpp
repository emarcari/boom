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

#ifndef BOOM_POISSON_BART_POSTERIOR_SAMPLER_HPP_
#define BOOM_POISSON_BART_POSTERIOR_SAMPLER_HPP_

#include <Models/Bart/PosteriorSamplers/BartPosteriorSampler.hpp>
#include <Models/Bart/PoissonBartModel.hpp>
#include <Models/Bart/ResidualRegressionData.hpp>
#include <Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp>

// The posterior sampler class defined in this file uses data
// augmentation to model Poisson errors in the BART model.  These
// comments describe the general data augmetnation strategy, which is
// equivalent to that in
// Models/Glm/PosteriorSamplers/PoissonRegressionAuxMixSampler.hpp.
// The sampling algorithm is described in a paper by
// Fruhwirth-Schnatter, Schnatter, Rue, and Held, "Improved Auxiliary
// Mixture Sampling for Hierarchical Models of Non-Gaussian Data."
// The paper appeared in "Statistics and Computing" (2009).
//
// A Poisson observation y is the number of events in the interval [0,
// E], where E is an exposure time (often set to 1 by default, but it
// need not be).  The events are produced by a Poisson process with
// rate lambda.  The complete data analysis of this process involves
// tracking a pair of events: the last event produced inside the
// interval, and the first event after the end of the interval.  Call
// the event times tau0 and tau1.  If y > 0 then tau0 is marginally
//
// [________|_______|___________________|______|______]_______|
// 0      event 1   2                   3     y=4     E    next_event
//
//

// Ga(y, lambda).  If y = 0 then tau0 = 0.  In either case tau1 - tau0
// is Exp(lambda).  Note that saying tau ~ Ga(y, lambda) is like
// saying (tau * lambda) ~ Ga(y, 1), which is independent of lambda.
// Now let z0 = -log(tau0), so z0 = log(lambda) - log(Ga(y, 1)), and
// let z1 = -log(tau1 - tau0), so z1 = log(lambda) - log(Exp(1).  Thus
// if we could see z0 (in the cases where it is finite) and z1 (which
// is always finite with probability 1), we would have two
// observations with which to estimate log(lambda).  If the Poisson
// observations are modeled using the typical "log link" then
// log(lambda) is the linear predictor, which in our case is the sum
// of trees.
//
// When it is time to impute tau0 given y, recall that the positions
// of homogeneous Poisson events are uniform in the interval.  Thus
// the position of the last event is a Beta(y, 1) random variable.
// This is a well known and easily proved result: Let M be the max of
// y independent uniform random variables.  Then Pr(M <= theta) =
// Pr(All <= theta) = theta^y.  The density of M is obtained by
// differentiating with respect to theta, so f_M(theta) = y *
// theta^{y-1}, which is a Beta(y, 1) distribution.
//
// Given imputed values for z0 and z1, their error distributions can
// be made Gaussian by approximating the distribution of -log(Ga(y,
// 1)) with a discrete mixture of normals.  This is done by a large
// table indexed by different values of y.  When z0 and z1 are imputed
// from y, their values are compared to the values of the mixture
// components for the appropriate distributions, and a value of "mu"
// and "sigma" is simultaneously imputed for each.  In this way, each
// observation is mapped to either 1 or 2 latent Gaussian observations
// with a "known" mean and variance.
namespace BOOM {

  namespace Bart {
    class PoissonSufficientStatistics;

    // The data augmentation trick for a Poisson is that
    class PoissonResidualRegressionData
        : public ResidualRegressionData {
     public:
      PoissonResidualRegressionData(Ptr<PoissonRegressionData> dp,
                                    double initial_predicted_log_lambda);
      int y()const;            // observed_response
      double exposure()const;  // observed exposure

      virtual void add_to_residual(double value);
      virtual void add_to_poisson_suf(PoissonSufficientStatistics &suf)const;

      // The term 'internal' refers to the largest observation inside
      // the interval [0, exposure).
      // Args:
      //   internal_residual:  -log(final_event_time) - mu[i]0.
      //   internal_weight:  1.0 / sigsq[i]0.
      //   external_residual:  -log(final_interarrival_time) - mu[i]1.
      //   external_weight:  1.0 / sigsq[i]1.
      // where mu[i] and sigsq[i] are the mean and variance of the
      // mixture component associated with observation i (0 for
      // internal and 1 for external).
      void set_latent_data(
          double internal_residual,
          double internal_weight,
          double external_residual,
          double external_weight);

      double internal_residual() const;
      double internal_weight() const;
      double external_residual() const;
      double external_weight() const;

      void set_predicted_log_lambda(double eta) {
        log_lambda_ = eta;
      }
      double predicted_log_lambda() const {return log_lambda_;}

     private:
      // Storing observed_data_ as a const raw pointer allows us to
      // avoid issues with reference counting in multi-threaded
      // programs.
      const PoissonRegressionData *observed_data_;

      // Let mu[i] and sigma^2[i] denote the mean and variance of the
      // mixture component associated with this observation and let
      // eta[i] denote the predicted value from the sum-of-trees.  The
      // internal residual is
      // negative_log_final_event_time - mu[i] - eta.
      // The internal_weight is 1.0 / sigsq[i].  The internal weight
      // is zero if this->y() == 0.
      double internal_residual_;
      double internal_weight_;

      // Let mu[i] and sigsq[i] denote the mean and variance of the
      // mixture component associated with the final_interarrival_time
      // for this observation.  These may be different from mu[i] and
      // sigsq[i] for the final_event_time.  The external_residual is
      // negative_log_final_interarrival_time - mu[i], and the
      // external weight is 1.0 / sigsq[i].
      double external_residual_;
      double external_weight_;

      // The log of the predicted value for the mean of this data
      // point.  This is the prediction made by the sum-of-trees.
      double log_lambda_;
    };

  //======================================================================
    // The sufficient statistics are the stats needed to compute log
    // integrated likelihood, and to draw the mean parameters at each
    // node.  A derivation of log integrated likelihood is given
    // immediately after the class definition.
    class PoissonSufficientStatistics
        : public SufficientStatisticsBase {
     public:
      virtual PoissonSufficientStatistics * clone()const;

      // Sets all data elements to 0.
      void clear();

      // Remember that all observations with y > 0 will have two
      // contributions to the sufficient statistics.
      virtual void update(const ResidualRegressionData &data);
      virtual void update(const PoissonResidualRegressionData &data);

      double sum_of_weights() const {return sum_of_weights_;}
      double weighted_sum_of_residuals() const {
        return weighted_sum_of_residuals_;
      }
     private:
      // The sum (over all data) of internal_weight()*internal_residual()
      // + external_weight()*external_residual();
      double weighted_sum_of_residuals_;

      // The sum (over all data) of internal_weight() +
      // external_weight().
      double sum_of_weights_;
    };

  /* Begin LaTeX documentation for the calculation of integrated log
   * likelihood, which reveals the structure of the complete data
   * sufficient statistics.
\documentclass{article}
\usepackage{amsmath}
\newcommand{\by}{{\bf y}}
\begin{document}
Suppose $y_i \sim N(\mu_i + \eta, \sigma^2_i)$, where $\mu_i$ and
$\sigma_i$ are known.  The prior is $\eta \sim N(\mu_0, \tau^2)$.  Let
$\by = (y_1, \dots, y_n)$, and $w_i = \sigma_i^{-2}$.  Let $v^{-1} =
\sum_i w_i + 1/\tau^2$ be the posterior precision of $\eta$, and let
$\tilde \mu = v(\sum_iw_i(y_i - \mu_i) + \mu_0/\tau^2)$ be the
posterior mean.  Then the integrated likelihood is
\begin{equation*}
  \begin{split}
    p(\by) & = \int p(\by | \eta) p(\eta) \ d \eta \\
    &= \int (2\pi)^{-n/2} \prod_i (w_i)^{1/2}
    \exp\left(-\frac{1}{2} \sum_i w_i [y_i - \mu_i - \eta]^2\right)\\
    & \qquad (2\pi)^{-1/2} (\tau^2)^{-1/2}
    \exp\left( -\frac{1}{2} [\eta - \mu_0]^2/\tau^2\right) \ d \eta \\
    & = \int (2\pi)^{-n+1/2} \prod_i w_i^{1/2} \tau^{-1}
    \exp \left(-\frac{1}{2} \left[
        \eta^2 \left(\sum_iw_i + \frac{1}{\tau^2}\right)
        -2 \eta \left(
          \sum_i w_i[y_i - \mu_i]
          + \frac{\mu_0}{\tau^2}\right)
      \right.
    \right. \\
    & \left. \left. \qquad + \sum_i w_i(y_i - \mu_i)^2 +
        \frac{\mu_0^2}{\tau^2} + \frac{\tilde \mu^2}{v} -
        \frac{\tilde\mu^2}{v} \right]
    \right) \ d \eta\\
    &= (2\pi)^{-n/2} \left(\prod_i w_i^{1/2}\right) \frac{v}{\tau}
    \exp\left(-\frac{1}{2}\left[
      \sum_i w_i(y_i - \mu_i)^2 +
        \frac{\mu_0^2}{\tau^2} - \frac{\tilde \mu^2}{v}\right]\right)
    \\
    & \qquad
    \int (2\pi)^{-1/2}\frac{1}{v}
    \exp\left( -\frac{1}{2v}(\eta - \tilde\mu)^2\right) \ d \eta \\
    &=(2\pi)^{-n/2} \left(\prod_i w_i^{1/2}\right) \frac{v}{\tau}
    \exp\left(-\frac{1}{2}\left[
      \sum_i w_i(y_i - \mu_i)^2 +
        \frac{\mu_0^2}{\tau^2} - \frac{\tilde \mu^2}{v}\right]\right).
  \end{split}
\end{equation*}
\end{document}
  ------ End LaTeX documentation ------------------------------------------*/
  } // namespace Bart

  class PoissonBartPosteriorSampler : public BartPosteriorSamplerBase {
   public:
    typedef Bart::PoissonResidualRegressionData DataType;
    typedef Bart::PoissonSufficientStatistics SufType;

    PoissonBartPosteriorSampler(PoissonBartModel *model,
                                double prior_mean_guess,
                                double prior_mean_sd,
                                double prior_tree_depth_alpha,
                                double prior_tree_depth_beta);

    virtual void draw();
    virtual double draw_mean(Bart::TreeNode *leaf);
    virtual double log_integrated_likelihood(
        const Bart::SufficientStatisticsBase *suf)const;

    virtual void clear_residuals();

    // The number of "residual data points" managed by the sampler.
    virtual int residual_size()const;
    virtual DataType * create_and_store_residual(int i);
    virtual SufType * create_suf() const;

    void impute_latent_data();
    void impute_latent_data_point(DataType *data);

   private:
    PoissonBartModel *model_;
    std::vector<boost::shared_ptr<DataType> > residuals_;
    boost::shared_ptr<PoissonDataImputer> data_imputer_;
  };

}  // namespace BOOM

#endif //  BOOM_POISSON_BART_POSTERIOR_SAMPLER_HPP_
