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

#include <Models/MvnBase.hpp>
#include <Models/Glm/MLogitSplit.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_split.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <LinAlg/Selector.hpp>
#include <cpputil/math_utils.hpp>
#include <distributions.hpp>
#include <stats/logit.hpp>
#include <cpputil/seq.hpp>

namespace BOOM{

  typedef MLVS_split MLVSS;
  typedef MlvsCdSuf_split SUF;

  SUF::MlvsCdSuf_split(uint xdim, uint Nch)
  {
    for(uint i=1; i<Nch; ++i){
      Spd S(xdim, 0.0);
      xtwx_.push_back(S);
      Vec x(xdim, 0.0);
      xtwu_.push_back(x);
    }
  }
  //----------------------------------------------------------------------
  SUF * SUF::clone()const{return new SUF(*this);}
  //----------------------------------------------------------------------
  void SUF::add(Ptr<SUF> s){
    uint n = xtwx_.size();
    for(uint i=0; i<n; ++i){
      xtwu_[i] += s->xtwu(i);
      xtwx_[i] += s->xtwx(i);
    }
    sym_ = false;
  }
  //----------------------------------------------------------------------
  void SUF::add(Ptr<MlvsCdSuf> s){ this->add(s.dcast<SUF>()); }
  //----------------------------------------------------------------------
  void SUF::clear(){
    uint n = xtwx_.size();
    for(uint i=0; i<n; ++i){
      xtwx_[i] = 0;
      xtwu_[i] = 0;
    }
    sym_ = false;
  }
  //----------------------------------------------------------------------
  void SUF::update(Ptr<ChoiceData> dp, const Vec & wgts, const Vec &u){
    sym_ = false;
    const Vec &x(dp->Xsubject());
    uint M = dp->nchoices();
    for(uint m=1; m<M; ++m){
      xtwx_[m-1].add_outer(x,wgts[m], false);
      xtwu_[m-1].axpy(x, wgts[m]*u[m]);
    }
  }
  //----------------------------------------------------------------------
  const Spd & SUF::xtwx(uint which)const{
    if(!sym_){
      for(uint i=0; i<xtwx_.size(); ++i) xtwx_[i].reflect();
      sym_ = true;
    }
    return xtwx_[which];
  }
  //----------------------------------------------------------------------
  const Vec & SUF::xtwu(uint which)const{ return xtwu_[which]; }


  //======================================================================

  void MLVSS::setup(){
    uint Nch = mod_->Nchoices();
    uint xdim = mod_->subject_nvars();
    uint beta_dim = xdim * (Nch-1);
    uint lo=0;
    for(uint i=1; i<Nch; ++i){
      Selector inc(beta_dim, false);
      for(uint j=0; j<xdim; ++j) inc.add(j+lo);
      lo+= xdim;
    }
  }

  //----------------------------------------------------------------------

  MLVSS::MLVS_split(MLogitSplit *m,
		    std::vector<Ptr<MvnBase> > b,
		    std::vector<Ptr<VSP> > v,
		    uint nthreads)
    : MLVS_base(m),
      mod_(m),
      bpri_(b),
      vpri_(v),
      suf(new SUF(mod_->subject_nvars(), mod_->Nchoices())),
      imp(new MlvsDataImputer(mod_, suf, nthreads)),
      Ominv(m->subject_nvars()),
      iV_tilde_(Ominv),
      beta_tilde_(m->subject_nvars())
  {
    setup();
  }
  //----------------------------------------------------------------------
  MLVSS::MLVS_split(MLogitSplit *m, Ptr<MvnBase> b, Ptr<VSP>  v,
		    uint nthreads)
    : MLVS_base(m),
      mod_(m),
      suf(new SUF(mod_->subject_nvars(), mod_->Nchoices())),
      imp(new MlvsDataImputer(mod_, suf, nthreads)),
      Ominv(m->subject_nvars()),
      iV_tilde_(Ominv),
      beta_tilde_(m->subject_nvars())
  {

    // stores copies of pointers to b and v, NOT pointers of copies.
    // this allows b and v to be modified externally.  If you want
    // distinct storage for b and v then use the ctor listed above

    uint nch = m->Nchoices();
    for(uint i=1; i<nch; ++i){
      bpri_.push_back(b);       // not cloned!!!
      vpri_.push_back(v);       // not cloned!!!
    }
    setup();
  }
  //----------------------------------------------------------------------
  double MLVSS::log_model_prob(const Selector &g, uint m){

    assert(m>0);
    uint which = m-1;
    double ans = vpri_[which]->logp(g);
    if(ans==BOOM::negative_infinity()) return ans;

    Ominv = g.select(bpri_[which]->siginv());
    double logdet = Ominv.logdet();
    if(logdet == BOOM::negative_infinity()) return logdet;

    bool ok=true;
    iV_tilde_ = Ominv + g.select(suf->xtwx(which));
    Mat L = iV_tilde_.chol(ok);
    if(!ok)  return BOOM::negative_infinity();
    logdet += 2*sum(log(L.diag()));

    Vec mu = g.select(bpri_[which]->mu());
    double Qform = Ominv.Mdist(mu);  // Qform = b^T Siginv b
    Vec S = g.select(suf->xtwu(which)) + Ominv * mu;
    Lsolve_inplace(L,S);
    Qform-= S.normsq();  // S.normsq =  beta_tilde ^T V_tilde beta_tilde

    ans += .5*(logdet - Qform);
    return ans;
  }
  //----------------------------------------------------------------------
  static inline bool keep_flip(double logp_old, double logp_new){
    double pflip = logit_inv(logp_new - logp_old);
    double u = runif(0,1);
    return u < pflip ? true : false;
  }
  //----------------------------------------------------------------------
  void MLVSS::draw_inclusion_vector(){
    uint M = mod_->Nchoices();
    for(uint m=1; m<M; ++m){
      Ptr<GlmCoefs> beta = mod_->Beta_subject_prm(m);
      Selector inc = beta->inc();
      uint nv = inc.nvars_possible();
      double logp = log_model_prob(inc,m);
      std::vector<uint> flips = seq<uint>(0,nv-1);
      std::random_shuffle(flips.begin(), flips.end());
      uint hi = std::min<uint>(nv, max_nflips());
      for(uint i=0; i<hi; ++i){
	uint I = flips[i];
	inc.flip(I);
	double logp_new = log_model_prob(inc,m);
	if( keep_flip(logp, logp_new)) logp = logp_new;
	else inc.flip(I);  // reject the flip, so flip back
      }
      mod_->set_inc_subject(inc,m);
    }
  }
  //----------------------------------------------------------------------
  double MLVSS::logpri()const{
    uint M = mod_->Nchoices();
    double ans=0;
    Vec b;
    for(uint m=1; m<M; ++m){
      Ptr<GlmCoefs> beta = mod_->Beta_subject_prm(m);
      uint which = m-1;
      const Selector &g(beta->inc());
      ans+= vpri_[which]->logp(g);
      if(g.nvars() > 0){
        Ptr<MvnBase> pri = bpri_[which];
        b = g.select(beta->beta());
        ans += dmvn(g.select(b), g.select(pri->mu()),
                    g.select(pri->siginv()), true);
      }
    }
    return ans;
  }
  //----------------------------------------------------------------------
  void MLVSS::impute_latent_data(){imp->draw();}
  //----------------------------------------------------------------------
  void MLVSS::draw_beta(){
    uint M = mod_->Nchoices();
    for(uint m = 1; m<M; ++m){
      uint which=m-1;
      Ptr<MvnBase> pri(bpri_[which]);
      Selector inc = mod_->Beta_subject_prm(m)->inc();
#ifndef NDEBUG
      uint n = inc.nvars();
      uint N = inc.nvars_possible();
      assert(n<=N);
#endif

      Ominv = inc.select(pri->siginv());
      Spd ivar = Ominv + inc.select(suf->xtwx(which));
      Vec b = inc.select(suf->xtwu(which)) + Ominv *inc.select(pri->mu());
      b = ivar.solve(b);
      Vec beta = rmvn_ivar(b,ivar);
      mod_->set_beta_subject(beta, m);
    }
  }
  //----------------------------------------------------------------------
  void MLVSS::find_posterior_mode(){

  }
  //----------------------------------------------------------------------
}
