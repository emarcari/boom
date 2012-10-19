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
#include "MultinomialLogitModel.hpp"
#include <cmath>

#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/lse.hpp>

#include <LinAlg/VectorView.hpp>
#include <numopt.hpp>
#include <TargetFun/Loglike.hpp>
#include <TargetFun/LogPost.hpp>

#include <boost/bind.hpp>

#include <Models/Glm/PosteriorSamplers/MLVS.hpp>
#include <numopt.hpp>
#include <Models/MvnBase.hpp>
#include <stats/FreqDist.hpp>

namespace BOOM{

  typedef MultinomialLogitModel MLM;
  typedef MLogitBase MLB;

  inline Vec make_vector(const Mat & beta_subject, const Vec &beta_choice){
     Vec b(beta_subject.begin(), beta_subject.end());
     b.concat(beta_choice);
     return b;
  }
  //------------------------------------------------------------
  void MLM::setup(){
    ParamPolicy::set_prm(new GlmCoefs(beta_size(false)));
    setup_observers();
    beta_with_zeros_current_=false;
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(uint Nch, uint Psub, uint Pch)
    : MLB(Nch,Psub,Pch),
      ParamPolicy()
  {
    setup();
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(const Mat & beta_subject, const Vec &beta_choice)
    : MLB(1 + beta_subject.ncol(), beta_subject.nrow(), beta_choice.size()),
      ParamPolicy()
  {
    setup();
    set_beta(make_vector(beta_subject,beta_choice));
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(
      const std::vector<Ptr<CategoricalData> > & responses,
      const Mat &Xsubject,
      const std::vector<Mat> &Xchoice)
    : MLB(responses, Xsubject, Xchoice),
      ParamPolicy()
  {
    setup();
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(const MultinomialLogitModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      MLB(rhs),
      ParamPolicy(rhs)
  {
    setup_observers();
  }
  //------------------------------------------------------------
  MLM * MLM::clone()const{return new MLM(*this);}
  //------------------------------------------------------------
  const Vec & MLM::beta()const{ return coef().Beta();}
  //------------------------------------------------------------
  const Vec & MLM::beta_with_zeros()const{
    if(!beta_with_zeros_current_) fill_extended_beta();
    return beta_with_zeros_;}
  //------------------------------------------------------------
  void MLM::set_beta(const Vec &b, bool reset_inc){
    coef().set_Beta(b, reset_inc);}

  //------------------------------------------------------------
  Vec MLM::beta_subject(uint choice)const{
    uint p = subject_nvars();
    if(choice==0) return Vec(p,0.0);
    const Vec &b(beta());
    Vec::const_iterator it = b.begin()+( (choice-1)*p);
    return Vec(it,it+p);
  }

  //------------------------------------------------------------
  void MLM::set_beta_subject(const Vec &b, uint m){
    if(m==0 || m>= Nchoices()) index_out_of_bounds(m);
    uint p = subject_nvars();
    Vec beta(this->beta());
    Vec::iterator it = beta.begin() + (m-1)*p;
    std::copy(b.begin(), b.end(), it);
    set_beta(beta);
  }
  //------------------------------------------------------------

  Vec MLM::beta_choice()const{
    Vec::const_iterator it = beta().begin();
    it+= (Nchoices()-1)*subject_nvars();
    return Vec(it, beta().end());
  }
  //------------------------------------------------------------

  void MLM::set_beta_choice(const Vec &b){
    uint pos = (Nchoices()-1)*subject_nvars();
    Vec beta(this->beta());
    std::copy(b.begin(), b.end(), beta.begin()+pos);
    set_beta(beta);
  }
  //------------------------------------------------------------
  GlmCoefs & MLM::coef(){return ParamPolicy::prm_ref();}
  const GlmCoefs & MLM::coef()const{return ParamPolicy::prm_ref();}
  Ptr<GlmCoefs> MLM::coef_prm(){return ParamPolicy::prm();}
  //------------------------------------------------------------
  const Ptr<GlmCoefs> MLM::coef_prm()const{return ParamPolicy::prm();}

  //------------------------------------------------------------
  Selector MLM::inc()const{ return coef().inc();}

  //------------------------------------------------------------

  double MLM::Loglike(Vec &g, Mat &h, uint nd)const{
    const std::vector<Ptr<ChoiceData> > & d(dat());
    double ans=0;
    if(nd>0){ g=0; if(nd>1) h=0; }
    uint n = d.size();
    Vec wsp(Nchoices());
    Vec xbar;
    Vec probs;
    Vec tmpx;
    Mat X;
    bool downsampling = log_sampling_probs().size()==Nchoices();
    Selector inc(this->inc());
    for(uint i=0; i<n; ++i){
      Ptr<ChoiceData> dp = d[i];
      uint y = dp->value();
      fill_eta(*dp,wsp);
      if(downsampling) wsp += log_sampling_probs();
      double lognc = lse(wsp);
      ans += wsp[y] - lognc;
      if(nd>0){
	uint M = dp->nchoices();
	X = inc.select_cols(dp->X(false));
	probs = exp(wsp - lognc);
	xbar = probs * X;
	g+= X.row(y) - xbar;

	if(nd>1){
	  for(uint m=0; m<M; ++m){
	    tmpx = X.row(m);
	    h.add_outer(tmpx, tmpx, -probs[m]);
	  }
	  h.add_outer(xbar,xbar);
	}
      }
    }
    return ans;
  }
  //----------------------------------------------------------------------
  void MLM::add_all_slopes(){
    coef().add_all();
  }
  //----------------------------------------------------------------------
  void MLM::drop_all_slopes(bool keep_int){
    coef().drop_all();
    if(keep_int){
      uint psub = subject_nvars();
      uint nch = Nchoices();
      for(uint m = 1; m<nch; ++m){
	uint pos = (m-1)*psub;
	coef().add(pos);
      }
    }
  }
  double MLM::predict_choice(Ptr<ChoiceData> dp, uint m)const{
    uint pch = choice_nvars();
    if(pch==0) return 0;
    uint psub = subject_nvars();
    uint M = Nchoices();
    ConstVectorView  b(subvector(beta(), (M-1)*psub));
    assert(b.size()== dp->choice_nvars());
    return b.dot(dp->Xchoice(m));
  }
  //------------------------------------------------------------
  double MLM::predict_subject(Ptr<ChoiceData> dp, uint m)const{
    if(m==0) return 0;
    uint psub = subject_nvars();
    assert(m < Nchoices());
    ConstVectorView b(subvector(beta(), (m-1)*psub, m*psub));
    return b.dot(dp->Xsubject());
  }
  //------------------------------------------------------------
  Vec MLM::eta(Ptr<ChoiceData> dp)const{
    Vec ans(Nchoices());
    return fill_eta(*dp, ans);
  }
  //------------------------------------------------------------

  Vec&  MLM::fill_eta(const ChoiceData & dp, Vec &ans)const{
    uint M = Nchoices();
    ans.resize(M);
    const Mat &X(dp.X());
    ans = X * beta_with_zeros();
    return ans;
  }

  //------------------------------------------------------------
  void MLM::setup_observers(){
    GlmCoefs & b(coef());
    try{
      b.add_observer(boost::bind(&MLM::watch_beta, this));
    } catch(const std::exception &e){
      throw_exception<std::runtime_error>(e.what());
    }catch(...){
      throw_exception<std::runtime_error>(
          "unknown exception (from boost::bind) caught by "
          "MultinomialLogitModel::setup_observer");
    }
  }

  //------------------------------------------------------------
  void MLM::watch_beta(){
    beta_with_zeros_current_ = false; }

  //------------------------------------------------------------
  void MLM::fill_extended_beta()const{
    uint p = subject_nvars();
    Vec &b(beta_with_zeros_);
    b.resize(beta_size(true));
    const Vec &Beta(beta());
    std::fill(b.begin(), b.begin()+p, 0);
    std::copy(Beta.begin(), Beta.end(), b.begin()+p);
    beta_with_zeros_current_ =true;
  }

  //------------------------------------------------------------
  void MLM::index_out_of_bounds(uint m)const{
    ostringstream err;
    err << "index " << m << " outside the allowable range (" << 1 << ", "
	<< Nchoices()-1 << ") in MultinomialLogitModel::set_beta_subject."
	<< endl;
    throw_exception<std::runtime_error>(err.str());
  }

  //______________________________________________________________________

  typedef MultinomialLogitEMC MLEMC;

  MLEMC::MultinomialLogitEMC(const Mat & beta_subject, const Vec &beta_choice)
    : MLM(beta_subject, beta_choice)
  {}

  MLEMC::MultinomialLogitEMC(uint Nchoices, uint subject_xdim, uint choice_xdim)
    : MLM(Nchoices, subject_xdim, choice_xdim)
  {}

  MultinomialLogitEMC * MLEMC::clone()const{return new MLEMC(*this);}


  double MLEMC::Loglike(Vec &g, Mat &h, uint nd)const{
    uint n = probs_.size();
    if(n==0) return MLM::Loglike(g,h,nd);
    const std::vector<Ptr<ChoiceData> > & d(dat());
    if(d.size()!=n){
      ostringstream err;
      err << "mismatch between data and probs_ in "
	  << "MultinomialLogitEMC::Loglike." <<endl
	;
      throw_exception<std::runtime_error>(err.str());
    }

    double ans=0;
    if(nd>0){ g=0; if(nd>1) h=0; }
    Vec wsp(Nchoices());
    Vec xbar;
    Vec probs;
    Vec tmpx;
    Mat X;
    bool downsampling = log_sampling_probs().size()==Nchoices();
    Selector inc(this->inc());
    for(uint i=0; i<n; ++i){
      double w = probs_[i];
      Ptr<ChoiceData> dp = d[i];
      uint y = dp->value();
      fill_eta(*dp,wsp);
      if(downsampling) wsp += log_sampling_probs();
      double lognc = lse(wsp);
      ans += w * (wsp[y] - lognc);
      if(nd>0){
	uint M = dp->nchoices();
	X = inc.select_cols(dp->X(false));
	probs = exp(wsp - lognc);
	xbar = probs * X;
	g.axpy(X.row(y) - xbar, w);

	if(nd>1){
	  for(uint m=0; m<M; ++m){
	    tmpx = X.row(m);
	    h.add_outer(tmpx, tmpx, -probs[m]*w);
	  }
	  h.add_outer(xbar,xbar, w);
	}
      }
    }
    return ans;
  }
  //----------------------------------------------------------------------
  void MLEMC::add_mixture_data(Ptr<Data> d, double prob){
    MLM::add_data(d);
    probs_.push_back(prob);
  }

  void MLEMC::clear_data(){
    MLM::clear_data();
    probs_.clear();
  }
  //----------------------------------------------------------------------

  void MLEMC::set_prior(Ptr<MvnBase> pri){
    pri_ = pri;
    uint dim = pri_->dim();
    NEW(VariableSelectionPrior, vpri)(dim);
    NEW(MLVS, sam)(this, pri_,vpri);
    set_method(sam);
  }

  void MLEMC::find_posterior_mode(){
    if(!pri_){
      ostringstream err;
      err << "MultinomialLogit_EMC cannot find posterior mode.  "
	  << "No prior is set." << endl;
      throw_exception<std::runtime_error>(err.str());
    }

    d2LoglikeTF loglike(this);
    d2LogPostTF logpost(loglike, pri_);
    Vec b = this->beta();
    uint dim = b.size();
    Vec g(dim);
    Mat h(dim,dim);
    b = max_nd2(b, g, h, Target(logpost), dTarget(logpost),
		   d2Target(logpost), 1e-5);
    this->set_beta(b);
  }
}
