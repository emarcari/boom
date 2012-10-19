/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include <Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp>
#include <cpputil/seq.hpp>
#include <distributions.hpp>

namespace BOOM{
  typedef BinomialLogitSpikeSlabSampler BLSSS;

  BLSSS::BinomialLogitSpikeSlabSampler(BinomialLogitModel *m,
                                       Ptr<MvnBase> pri,
                                       Ptr<VariableSelectionPrior> vpri,
                                       int clt_threshold)
      : BinomialLogitSampler(m, pri, clt_threshold),
        m_(m),
        pri_(pri),
        vpri_(vpri),
        allow_model_selection_(true),
        max_flips_(-1)
  {}

  void BLSSS::draw(){
    impute_latent_data();
    if(allow_model_selection_) draw_model_indicators();
    draw_beta();
  }

  void BLSSS::draw_beta(){
    Selector g = m_->coef().inc();
    if(g.nvars() == 0){
      Vec b(g.nvars_possible(), 0.0);
      m_->set_Beta(b, allow_model_selection_);
      return;
    }
    ivar_ = g.select(pri_->siginv());
    ivar_mu_ = ivar_ * g.select(pri_->mu());
    ivar_ += g.select(xtwx());
    ivar_mu_ += g.select(xtwu());
    Vec b = ivar_.solve(ivar_mu_);
    b = rmvn_ivar(b, ivar_);

    // If model selection is turned off and some elements of beta
    // happen to be zero (because, e.g., of a failed MH step) we don't
    // want the dimension of beta to change.
    m_->set_Beta(g.expand(b), allow_model_selection_);
  }

  double BLSSS::logpri()const{
    const Selector & g(m_->coef().inc());
    double ans = vpri_->logp(g);  // p(gamma)
    if(ans == BOOM::infinity(-1)) return ans;
    if(g.nvars() > 0){
      ans += dmvn(m_->beta(),
                  g.select(pri_->mu()),
                  g.select(pri_->siginv()),
                  true);
    }
    return ans;
  }

  double BLSSS::log_model_prob(const Selector &g)const{
    // borrowed from MLVS.cpp
    double num = vpri_->logp(g);
    if(num==BOOM::infinity(-1) || g.nvars() == 0){
      // If num == -infinity then it is in a zero support point in the
      // prior.  If g.nvars()==0 then all coefficients are zero
      // because of the point mass.  The only entries remaining in the
      // likelihood are sums of squares of y[i] that are independent
      // of g.  They need to be omitted here because they are omitted
      // in the non-empty case below.
      return num;
    }
    ivar_ = g.select(pri_->siginv());
    num += .5*ivar_.logdet();
    if(num == BOOM::infinity(-1)) return num;

    Vec mu = g.select(pri_->mu());
    ivar_mu_ = ivar_ * mu;
    num -= .5*mu.dot(ivar_mu_);

    bool ok=true;
    ivar_ = ivar_ + g.select(xtwx());
    Mat L = ivar_.chol(ok);
    if(!ok)  return BOOM::infinity(-1);
    double denom = sum(log(L.diag()));  // = .5 log |ivar_|
    Vec S = g.select(xtwu()) + ivar_mu_;
    Lsolve_inplace(L,S);
    denom-= .5*S.normsq();  // S.normsq =  beta_tilde ^T V_tilde beta_tilde
    return num-denom;
  }

  void BLSSS::allow_model_selection(bool tf){
    allow_model_selection_ = tf;
  }

  void BLSSS::limit_model_selection(int max_flips){
    max_flips_ = max_flips;
  }

  void BLSSS::draw_model_indicators(){
    Selector g = m_->coef().inc();
    std::vector<int> indx = seq<int>(0, g.nvars_possible()-1);
    std::random_shuffle(indx.begin(), indx.end());
    double logp = log_model_prob(g);

    if(!finite(logp)){
      vpri_->make_valid(g);
      logp = log_model_prob(g);
    }
    if(!finite(logp)){
      ostringstream err;
      err << "BinomialLogitSpikeSlabSampler did not start with a legal configuration."
          << endl << "Selector vector:  " << g << endl
          << "beta: " << m_->beta() << endl;
      report_error(err.str());
    }

    uint n = g.nvars_possible();
    if(max_flips_ > 0) n = std::min<int>(n, max_flips_);
    for(uint i=0; i<n; ++i){
      logp = mcmc_one_flip(g, indx[i], logp);
    }
    m_->coef().set_inc(g);
  }

  double BLSSS::mcmc_one_flip(Selector &mod, uint which_var, double logp_old){
    mod.flip(which_var);
    double logp_new = log_model_prob(mod);
    double u = runif(0,1);
    if(log(u) > logp_new - logp_old){
      mod.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }

}
