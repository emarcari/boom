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
#include "MLVS.hpp"

#include <sstream>
#include <cmath>
#include <algorithm>

#include <distributions.hpp>       // for rlexp,dnorm,rmvn
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/MvnBase.hpp>
#include <Models/Glm/ChoiceData.hpp>
#include <stats/logit.hpp>
#include <stdexcept>
#include <cpputil/math_utils.hpp>
#include <cpputil/seq.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{
  typedef MultinomialLogitModel MLM;
  typedef MlvsCdSuf_ml MLVSS;
  using std::ostringstream;


  MLVS::MLVS(MLM *Mod, Ptr<MvnBase> Pri,
	     Ptr<VariableSelectionPrior> Vpri,
	     uint nthreads, bool check_initial_condition)
    : MLVS_base(Mod),
      mod_(Mod),
      pri(Pri),
      vpri(Vpri),
      suf(new MLVSS(Mod->beta_size(false))),
      imp(new MlvsDataImputer(mod_, suf, nthreads))
  {
    if(check_initial_condition){
      if(!BOOM::finite(this->logpri())){
	ostringstream err;
	err << "MLVS initialized with an a priori illegal value" << endl
	    << "the initial Selector vector was: " << endl
	    << mod_->coef()->inc() << endl
	    << *vpri << endl;

	throw_exception<std::runtime_error>(err.str());
      }
    }

  }

  //______________________________________________________________________
  // public interface


  void MLVS::impute_latent_data(){imp->draw();}

  double MLVS::logpri()const{
    const Selector &g = mod_->coef()->inc();
    double ans = vpri->logp(g);
    if(ans==BOOM::infinity(-1)) return ans;
    if(g.nvars() > 0){
      ans += dmvn(g.select(mod_->beta()),
                  g.select(pri->mu()),
                  g.select(pri->siginv()),
                  true);
    }
    return ans;
  }

  //______________________________________________________________________
  // Drawing parameters

  MLVSS::MlvsCdSuf_ml(uint dim)
    : xtwx_(dim),
      xtwu_(dim),
      sym_(false)
  {}

  MLVSS * MLVSS::clone()const{ return new MLVSS(*this);}

  void MLVSS::clear(){
    xtwx_ = 0;
    xtwu_ = 0;
    sym_ = false;
  }

  void MLVSS::update(Ptr<ChoiceData> dp, const Vec & wgts, const Vec &u){
    const Mat & X(dp->X(false));      // 'false' means omit columns
    xtwx_.add_inner(X, wgts,false);   // corresponding to subject X's at
    xtwu_ += X.Tmult(wgts*u);         // choice level 0.
    sym_ = false;
  }

  void MLVSS::add(Ptr<MlvsCdSuf> s){
    Ptr<MlvsCdSuf_ml> news(s.dcast<MlvsCdSuf_ml>());
    if(!news){
      ostringstream err;
      err << "could not convert MlvsCdSuf to proper type in MLVS.cpp"
	  << endl;
      throw_exception<std::runtime_error>(err.str());
    }
    this->add(news);
  }

  void MLVSS::add(Ptr<MlvsCdSuf_ml> s){
    xtwx_ += s->xtwx();
    xtwu_ += s->xtwu();
    sym_ = false;
  }

  const Spd & MLVSS::xtwx()const{
    if(!sym_) xtwx_.reflect();
    sym_ = true;
    return xtwx_;
  }

  const Vec & MLVSS::xtwu()const{return xtwu_;}

  //======================================================================

  void MLVS::draw_beta(){
    const Selector  &inc(mod_->coef()->inc());
    Spd Ominv = inc.select(pri->siginv());
    Spd ivar = Ominv + inc.select(suf->xtwx());
    Vec b = inc.select(suf->xtwu()) + Ominv *inc.select(pri->mu());
    b = ivar.solve(b);
    Vec beta = rmvn_ivar(b,ivar);
    uint N = inc.nvars_possible();
    uint n = b.size();
    Vec Beta(N, 0.);
    for(uint i=0; i<n; ++i){
      uint I = inc.indx(i);
      Beta[I] = beta[i];
    }
    mod_->set_beta(Beta);
  }

  inline bool keep_flip(double logp_old, double logp_new){
    if(!finite(logp_new)) return false;
    double pflip = logit_inv(logp_new - logp_old);
    double u = runif(0,1);
    return u < pflip ? true : false;
  }

  void MLVS::draw_inclusion_vector(){
    Selector inc = mod_->coef()->inc();
    uint nv = inc.nvars_possible();
    double logp = log_model_prob(inc);
    if(!finite(logp)){
      logp = log_model_prob(inc);
      ostringstream err;
      err << "MLVS did not start with a legal configuration." << endl
	  << "Selector vector:  " << inc << endl
	  << "beta: " << mod_->beta() << endl;
      throw_exception<std::runtime_error>(err.str());
    }

    std::vector<uint> flips = seq<uint>(0, nv-1);
    std::random_shuffle(flips.begin(), flips.end());
    uint hi = std::min<uint>(nv, max_nflips());
    for(uint i=0; i<hi; ++i){
      uint I = flips[i];
      inc.flip(I);
      double logp_new = log_model_prob(inc);
      if( keep_flip(logp, logp_new)) logp = logp_new;
      else inc.flip(I);  // reject the flip, so flip back
    }
    mod_->coef()->set_inc(inc);
  }

  //______________________________________________________________________
  // computing probabilities

  double MLVS::log_model_prob(const Selector & g){
    double num = vpri->logp(g);
    if(num==BOOM::infinity(-1)) return num;

    Ominv = g.select(pri->siginv());
    num += .5*Ominv.logdet();
    if(num == BOOM::infinity(-1)) return num;

    Vec mu = g.select(pri->mu());
    Vec Ominv_mu = Ominv * mu;
    num -= .5*mu.dot(Ominv_mu);

    bool ok=true;
    iV_tilde_ = Ominv + g.select(suf->xtwx());
    Mat L = iV_tilde_.chol(ok);
    if(!ok)  return BOOM::infinity(-1);
    double denom = sum(log(L.diag()));  // = .5 log |Ominv|

    Vec S = g.select(suf->xtwu()) + Ominv_mu;
    Lsolve_inplace(L,S);
    denom-= .5*S.normsq();  // S.normsq =  beta_tilde ^T V_tilde beta_tilde

    return num-denom;
  }

}
