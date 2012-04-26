/*
  Copyright (C) 2005 Steven L. Scott

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
#include <Models/Glm/MLogitSplit.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <cmath>
#include <cpputil/math_utils.hpp>
#include <cpputil/lse.hpp>

#include <numopt.hpp>
#include <TargetFun/LoglikeSubset.hpp>
#include <boost/bind.hpp>

#include <LinAlg/SubMatrix.hpp>
#include <LinAlg/Array.hpp>

namespace BOOM{

  typedef MLogitSplit MLS;
  typedef MLogitBase MLB;
  //------------------------------------------------------------
  void MLS::setup_beta_subject(){
    uint nch = Nchoices();
    uint psub = subject_nvars();
    for(uint m=0; m<nch-1; ++m){
      NEW(GlmCoefs, beta)(psub);
      beta_subject_.push_back(beta);
    }
  }
  //------------------------------------------------------------
  void MLS::setup_params(){
    for(uint i=0; i<beta_subject_.size(); ++i)
      ParamPolicy::add_params(beta_subject_[i]);
    if(choice_nvars()>0) ParamPolicy::add_params(beta_choice_);
  }
  //------------------------------------------------------------
  MLS::MLogitSplit(const Mat & beta_subject, const Vec &beta_choice)
    : MLB(beta_subject.ncol()+1, beta_subject.nrow(), beta_choice.size()),
      ParamPolicy(),
      beta_subject_(beta_subject.ncol()),
      beta_choice_(new GlmCoefs(beta_choice))
  {
    for(uint i=0; i<beta_subject.ncol(); ++i){
      Vec bs(beta_subject.col_begin(i), beta_subject.col_end(i));
      NEW(GlmCoefs, beta)(bs);
      beta_subject_.push_back(beta);
    }
    setup_params();
  }
  //------------------------------------------------------------
  MLS::MLogitSplit(ResponseVec responses,
		   const Mat &Xsubject,
		   const Array &Xchoice)
    : MLB(responses, Xsubject, Xchoice),
      ParamPolicy(),
      beta_choice_(new GlmCoefs(Xchoice.dim(2)))
  {
    setup_beta_subject();
    setup_params();
  }
  //------------------------------------------------------------
  MLS::MLogitSplit(ResponseVec responses, const Mat &Xsubject)
    : MLB(responses, Xsubject),
      ParamPolicy(),
      beta_choice_(new GlmCoefs(0) )
  {
    setup_beta_subject();
    setup_params();
  }
  //------------------------------------------------------------
  MLS::MLogitSplit(const std::vector<Ptr<ChoiceData> >  &dv)
    : MLB(dv),
      ParamPolicy(),
      beta_choice_(new GlmCoefs(dv[0]->choice_nvars()))
  {
    setup_beta_subject();
    setup_params();
  }
  //------------------------------------------------------------
  MLS::MLogitSplit(uint Nch, uint Psub, uint Pch)
    : MLB(Nch,Psub,Pch)
  {
    setup_beta_subject();
    setup_params();
  }
  //------------------------------------------------------------
  MLS::MLogitSplit(const MLogitSplit &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      MLB(rhs),
      ParamPolicy(rhs),
      beta_choice_(rhs.beta_choice_->clone())
  {

    for(uint i=0; i<rhs.beta_subject_.size(); ++i){
      Ptr<GlmCoefs> beta(rhs.beta_subject_[i]->clone());
      beta_subject_.push_back(beta);
    }
    setup_params();
  }
  //------------------------------------------------------------
  MLS * MLS::clone()const{return new MLS(*this);}

  //----------------------------------------------------------------------
  void 	MLS::update_grad(const Vec & probs, Ptr<ChoiceData> dp,
			 Vec & g, Vec & wbar,
			 std::vector<Vec> &x,
			 std::vector<Vec> & w)const{
    uint M = Nchoices();
    uint y = dp->value();
    uint lo = 0;
    for(uint m=1; m<M; ++m){
      x[m] = Beta_subject_prm(m)->inc().select(dp->Xsubject());
      uint sz = x[m].size();
      double I = y==m ? 1:0;
      VectorView tmpg(g, lo, sz);
      tmpg.axpy(x[m],I-probs[m]);
      lo+= sz;
    }
    if(choice_nvars()>0){
      wbar=0;
      Selector inc(Beta_choice_prm()->inc());
      for(uint m=0; m<M; ++m){
	w[m] = inc.select(dp->Xchoice(m));
	wbar.axpy(w[m], probs[m]);
      }
      VectorView tmpg(g,lo, wbar.size());
      tmpg += w[y]-wbar;
    }
  }
  //----------------------------------------------------------------------
  void MLS::update_hess(const Vec & probs, const Vec & wbar,
			Mat &h, const std::vector<Vec> & x,
			const std::vector<Vec> & w)const{
    uint lo1 = 0;
    uint M = Nchoices();
    uint pch = choice_nvars();

    for(uint m=1; m<M; ++m){
      // step 1:  do the beta sub-blocks
      const Vec & x1(x[m]);
      uint sz1 = x1.size();
      uint lo2 = lo1;
      for(uint k=m; k<M; ++k){
	const Vec & x2(x[k]);
	uint sz2 = x2.size();
	Mat tmph(sz1, sz2);
	double I = (m==k) ? 1:0;
	tmph.add_outer(x1,x2, -(I-probs[m])*probs[k]);
	SubMatrix block(h, lo1, lo1+sz1-1, lo2, lo2+sz2-1);
	block += tmph;
	if(k>m){
	  SubMatrix lower_block(h,lo2,lo2+sz2-1, lo1, lo1+sz1-1);
	  lower_block+= tmph.t();
	}
	lo2+=sz2;
      }
      if(pch>0){  // step 2:  now do the cross-derivative
	uint sz2 = wbar.size();
	SubMatrix right_block(h, lo1, lo1 + sz1-1, lo2, lo2 + sz2-1);
	Mat tmph(sz1, sz2);
	tmph.add_outer(x[m], w[m]-wbar,-probs[m]);
	right_block+= tmph;
	SubMatrix lower_block(h, lo2, lo2+sz2-1, lo1, lo1+sz1-1);
	lower_block+= tmph.t();
      }
      lo1+=sz1;
    }
    // step 3:  now add in lower right corner.. the choice second derivative
    if(pch>0){
      uint nc = h.ncol();
      SubMatrix block(h, lo1, nc-1, lo1, nc-1);
      Spd tmph(nc-lo1);
      tmph.add_outer(wbar, -1);
      for(uint m=0; m<M; ++m) tmph.add_outer(w[m], probs[m]);
      block+= tmph;
    }
  }

  //------------------------------------------------------------
  double MLS::Loglike(Vec &g, Mat &h, uint nd)const{
    const std::vector<Ptr<ChoiceData> > & d(dat());
    double ans=0;
    uint n = d.size();
    Vec wsp(Nchoices());
    Vec wbar;
    bool downsampling = log_sampling_probs().size()==Nchoices();
    std::vector<Vec> ww;
    std::vector<Vec> xx;
    if(nd>0){
      g=0;
      uint M = Nchoices();
      xx.resize(M);
      ww.resize(M);
      if(choice_nvars()>0) wbar.resize(Beta_choice_prm()->inc().nvars());
      if(nd>1) 	h=0;
    }

    for(uint i=0; i<n; ++i){
      Ptr<ChoiceData> dp = d[i];
      uint y = dp->value();
      fill_eta(*dp,wsp);
      if(downsampling) wsp += log_sampling_probs();
      double lognc = lse(wsp);
      ans += wsp[y] - lognc;
      if(nd>0){
	Vec probs(exp(wsp - lognc));
	update_grad(probs, dp, g, wbar, xx, ww);
	if(nd>1) update_hess(probs, wbar, h,xx,ww);
      }
    }
    return ans;
  }
  //------------------------------------------------------------
  Ptr<GlmCoefs> MLS::Beta_choice_prm(){return beta_choice_;}
  Ptr<GlmCoefs> MLS::Beta_subject_prm(uint i){assert(i>0); return beta_subject_[i-1];}
  const Ptr<GlmCoefs> MLS::Beta_choice_prm()const{return beta_choice_;}
  const Ptr<GlmCoefs> MLS::Beta_subject_prm(uint i)const{assert(i>0); return beta_subject_[i-1];}
  //------------------------------------------------------------
  double MLS::predict_choice(const ChoiceData & dp, uint m)const{
    return beta_choice_->predict(dp.Xchoice(m));   }
  //------------------------------------------------------------
  double MLS::predict_subject(const ChoiceData & dp, uint m)const{
    if(m==0) return 0;
    return beta_subject_[m-1]->predict(dp.Xsubject());}
  //------------------------------------------------------------
  Vec MLS::eta(Ptr<ChoiceData> dp)const{
    Vec ans(Nchoices());
    return fill_eta(*dp, ans);
  }
  //------------------------------------------------------------
  Vec&  MLS::fill_eta(const ChoiceData & dp, Vec &ans)const{
    uint M = Nchoices();
    ans.resize(M);
    for(uint m=0; m<M; ++m)
      ans[m]= predict_subject(dp,m) + predict_choice(dp,m);
    return ans;
  }
  //------------------------------------------------------------
  Vec MLS::beta_subject(uint choice)const{
    if(choice==0) return Vec(subject_nvars(), 0.0);
    return beta_subject_[choice-1]->Beta(); }
  //------------------------------------------------------------
  Vec MLS::beta_choice()const{return beta_choice_->Beta(); }
  //------------------------------------------------------------
  void MLS::set_beta_choice(const Vec &b, bool infer_inc){
    uint nb = b.size();
    if(nb==beta_choice_->nvars())
      beta_choice_->set_beta(b);
    else if(nb==beta_choice_->nvars_possible()){
      beta_choice_->set_Beta(b, infer_inc);
    }else{
      ostringstream err;
      err << "wrong dimension for beta in MLS::set_beta_choice" << endl
	  << "dim(b) = " << b.size() << endl
	  << "nvars  = " << beta_choice_->nvars() << endl
	  << "nvars_possible = " << beta_choice_->nvars_possible() << endl;
      throw_exception<std::runtime_error>(err.str());
    }
  }
  //------------------------------------------------------------
  void MLS::set_beta_subject(const Vec &b, uint m, bool infer_inc){
    assert(m>0 && m < Nchoices());
    uint nb = b.size();
    Ptr<GlmCoefs> B(beta_subject_[m-1]);
    if(nb==B->nvars()) B->set_beta(b);
    else if(nb==B->nvars_possible()){
      B->set_Beta(b, infer_inc);
    }else{
      ostringstream err;
      err << "wrong dimension for beta in MLS::set_beta_subject" << endl
	  << "dim(b) = " << b.size() << endl
	  << "nvars  = " << B->nvars() << endl
	  << "nvars_possible = " << B->nvars_possible() << endl
	  << "m = " << m << endl;
      throw_exception<std::runtime_error>(err.str());
    }
  }
  //------------------------------------------------------------
  void MLS::set_inc_subject(const Selector &inc, uint m){
    assert(m>0 && m < Nchoices());
    Ptr<GlmCoefs> B(beta_subject_[m-1]);
#ifndef NDEBUG
    uint ni = inc.nvars_possible();
    uint nB = B->nvars_possible();
    assert(ni==nB);
#endif
    B->set_inc(inc);
  }
  //------------------------------------------------------------
  void MLS::set_inc_choice(const Selector &inc){
    assert(inc.nvars_possible()==
	   beta_choice_->nvars_possible());
    beta_choice_->set_inc(inc);
  }
  //------------------------------------------------------------
  void MLS::add_all_slopes(){
    uint M = Nchoices();
    for(uint m=1; m<M; ++m)
      Beta_subject_prm(m)->add_all();
    if(choice_nvars()>0) Beta_choice_prm()->add_all();
  }
  //------------------------------------------------------------
  void MLS::drop_all_slopes(bool keep_int){
    uint M = Nchoices();
    for(uint m=1; m<M; ++m){
      Ptr<GlmCoefs> beta = Beta_subject_prm(m);
      beta->drop_all();
      if(keep_int) beta->add(0);
    }
    if(choice_nvars()>0){
      Beta_choice_prm()->drop_all();
    }
  }

}
