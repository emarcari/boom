/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_BREG_VS_SAMPLER_HPP
#define BOOM_BREG_VS_SAMPLER_HPP
#include <Models/Glm/RegressionModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/GlmMvnPrior.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <Models/MvnGivenSigma.hpp>
#include <Models/GammaModel.hpp>

namespace BOOM{

  class BregVsSampler : public PosteriorSampler{
    // prior:  beta | gamma, sigma ~ Normal(b, sigma^2 * Omega)
    //                   1/sigma^2 ~ Gamma(sigma.guess, df)
    //                       gamma ~ VsPrior (product of Bernoulli)

    // A good choice for Omega^{-1} is kappa * XTX/n, which is kappa
    // 'typical' observations.

  public:

    BregVsSampler(Ptr<RegressionModel>,
		  double prior_nobs,     // Omega is prior_nobs * XTX/n
		  double expected_rsq,   // sigsq_guess = sample variance * this
		  double expected_model_size);  // prior inclusion probs = this/dim

    BregVsSampler(Ptr<RegressionModel>, const Vec & b,
		  const Spd & Omega_inverse,
		  double sigma_guess, double df,
		  const Vec &prior_inclusion_probs);

    BregVsSampler(Ptr<RegressionModel> m,
		  Ptr<GlmMvnPrior> bpri,
		  Ptr<GammaModel> sinv_pri,
		  Ptr<VariableSelectionPrior> vpri);

    virtual void draw();
    virtual double logpri()const;
    double log_model_prob(const Selector &inc)const;
    void supress_model_selection();
    void allow_model_selection();
    void supress_beta_draw();
    void supress_sigma_draw();
    void allow_sigma_draw();
    void allow_beta_draw();
    void limit_model_selection(uint nflips);

    double prior_df()const;
    double prior_ss()const;
  private:
    Ptr<RegressionModel> m_;

    Ptr<GlmMvnPrior> bpri_;  // mean is b_, ivar is ominv_

    Ptr<GammaModel> spri_;
    Ptr<VariableSelectionPrior> vpri_;

    std::vector<uint> indx;
    uint max_nflips_;
    bool draw_beta_;
    bool draw_sigma_;

    mutable Vec beta_tilde_;      // this is work space for computing
    mutable Spd iV_tilde_;        // posterior model probs
    mutable double DF_, SS_;

    double set_reg_post_params(const Selector &g, bool do_ldoi)const;
    double mcmc_one_flip(Selector &g, uint which_var, double logp_of_g);

    void draw_beta();
    void draw_model_indicators();
    void draw_sigma();

  };

}
#endif// BOOM_BREG_VS_SAMPLER_HPP
