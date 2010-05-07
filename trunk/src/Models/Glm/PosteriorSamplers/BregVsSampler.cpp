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

#include <Models/Glm/PosteriorSamplers/BregVsSampler.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/seq.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef BregVsSampler BVS;

  BVS::BregVsSampler(Ptr<RegressionModel> mod,
		     double prior_nobs,
		     double expected_rsq,
		     double expected_model_size)
    : m_(mod),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true)
  {
    uint p = m_->nvars_possible();
    Vec b = Vec(p, 0.0);
    Spd ominv(m_->suf()->xtx());
    double n = m_->suf()->n();
    ominv *= prior_nobs/n;

    bpri_ = new GlmMvnPrior(b, ominv, true);

    double v = m_->suf()->SST()/(n-1);
    assert(expected_rsq > 0 && expected_rsq < 1);
    double sigma_guess = v * (1-expected_rsq);

    spri_ = new GammaModel(prior_nobs/2, pow(sigma_guess, 2)*prior_nobs/2);

    double prob = expected_model_size/p;
    if(prob>1) prob = 1.0;
    Vec pi(p, prob);
    pi[0] = 1.0;

    vpri_ = new VariableSelectionPrior(pi);

  }

  BVS::BregVsSampler(Ptr<RegressionModel> mod, const Vec & b,
		const Spd & Omega_inverse,
		double sigma_guess, double df,
		const Vec &prior_inclusion_probs)
    : m_(mod),
      bpri_(new GlmMvnPrior(b, Omega_inverse, true)),
      spri_(new GammaModel(df/2, pow(sigma_guess, 2)*df/2)),
      vpri_(new VariableSelectionPrior(prior_inclusion_probs)),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true)
  {}

  BVS::BregVsSampler(Ptr<RegressionModel> mod,
		     Ptr<GlmMvnPrior> bpri,
		     Ptr<GammaModel> spri,
		     Ptr<VariableSelectionPrior> vpri)
    : m_(mod),
      bpri_(bpri),
      spri_(spri),
      vpri_(vpri),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true)
  {}

  void BVS::limit_model_selection(uint n){ max_nflips_ =n;}
  void BVS::allow_model_selection(){ max_nflips_ = indx.size();}
  void BVS::supress_model_selection(){max_nflips_ =0;}
  void BVS::supress_beta_draw(){draw_beta_ = false;}
  void BVS::allow_beta_draw(){draw_beta_ = false;}
  void BVS::supress_sigma_draw(){draw_sigma_ = false;}
  void BVS::allow_sigma_draw(){draw_sigma_ = false;}

  double BVS::prior_df()const{ return spri_->alpha()/2.0; }
  double BVS::prior_ss()const{ return spri_->beta()/2.0;}

  double BVS::log_model_prob(const Selector &g)const{
    if(g.nvars()==0) return BOOM::infinity(-1);
    double ldoi = set_reg_post_params(g, true);
    double ans = vpri_->logp(g)+ .5*(ldoi - iV_tilde_.logdet());
    ans -=  (.5*DF_-1)*log(SS_);
    return ans;
  }


  double BVS::mcmc_one_flip(Selector &mod, uint which_var, double logp_old){
    mod.flip(which_var);
    double logp_new = log_model_prob(mod);
    double u = runif(0,1);
    if(log(u) > logp_new - logp_old){
      mod.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }

  void BVS::draw(){
    if(max_nflips_>0) draw_model_indicators();
    if(draw_beta_ || draw_sigma_){
      set_reg_post_params(m_->coef()->inc(),false);
    }
    if(draw_sigma_) draw_sigma();
    if(draw_beta_) draw_beta();
  }

  void BVS::draw_sigma(){
    double siginv = rgamma(DF_/2.0, SS_/2.0);
    m_->set_sigsq(1.0/siginv);
  }

  void BVS::draw_beta(){
    iV_tilde_ /= m_->sigsq();
    beta_tilde_ = rmvn_ivar(beta_tilde_, iV_tilde_);
    m_->set_beta(beta_tilde_);
  }

  void BVS::draw_model_indicators(){
    Selector g = m_->coef()->inc();
    std::random_shuffle(indx.begin(), indx.end());
    double logp = log_model_prob(g);

    if(!finite(logp)){
      ostringstream err;
      err << "BregVsSampler did not start with a legal configuration." << endl
	  << "Selector vector:  " << g << endl
	  << "beta: " << m_->beta() << endl;
      throw std::runtime_error(err.str());
    }

    uint n = std::min<uint>(max_nflips_, g.nvars_possible());
    for(uint i=0; i<n; ++i){
      logp = mcmc_one_flip(g, indx[i], logp);
    }
    m_->coef()->set_inc(g);
  }


  double BVS::logpri()const{
    double ans = vpri_->logp(m_->coef()->inc());  // p(gamma)
    if(ans == BOOM::infinity(-1)) return ans;

    double sigsq = m_->sigsq();
    ans += spri_->logp(1.0/sigsq);               // p(1/sigsq)
    Selector g = m_->coef()->inc();
    ans += dmvn(m_->beta(), g.select(bpri_->mu()),
		g.select(bpri_->siginv())/sigsq, true);
    return ans;
  }

  double BVS::set_reg_post_params(const Selector &g, bool do_ldoi)const{
    Vec b = g.select(bpri_->mu());

    Spd Ominv = g.select(bpri_->siginv());
    double ldoi(0);
    if(do_ldoi) ldoi = Ominv.logdet();

    Spd xtx = m_->suf()->xtx(g);
    Vec xty = m_->suf()->xty(g);

    iV_tilde_ = Ominv + xtx;
    beta_tilde_ = Ominv * b + xty;
    beta_tilde_ = iV_tilde_.solve(beta_tilde_);

    DF_ = m_->suf()->n() + prior_df();
    SS_ = prior_ss() + m_->suf()->yty() + Ominv.Mdist(b);
    SS_ -= iV_tilde_.Mdist(beta_tilde_);
    return ldoi;
  }



}
