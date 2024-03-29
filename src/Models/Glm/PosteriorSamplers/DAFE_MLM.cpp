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

#include "DAFE_MLM.hpp"
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/MvnModel.hpp>
#include <Models/MvtModel.hpp>
#include <cmath>
#include <cpputil/math_utils.hpp>  // for lse
#include <cpputil/lse.hpp>
#include <distributions.hpp>       // for rlexp

#include <LinAlg/VectorView.hpp>
#include <TargetFun/TargetFun.hpp>
#include <TargetFun/LoglikeSubset.hpp>
#include <TargetFun/LogPost.hpp>

namespace BOOM{

  typedef MultinomialLogitModel MLM;
  typedef MetropolisHastings MH;


  //------------------------------------------------------------

  DafeMlmBase::DafeMlmBase(MultinomialLogitModel *mod,
			   Ptr<MvnModel> SubjectPri,  // each subject beta has this prior
			   Ptr<MvnModel> ChoicePri,
			   bool draw_b0)
    : mlm_(mod),
      subject_pri_(SubjectPri),
      choice_pri_(ChoicePri),
      mlo_(draw_b0 ? 0 : 1)
  {
    compute_xtx();
  }
  //------------------------------------------------------------
  double DafeMlmBase::logpri()const{
    uint M = mlm_->Nchoices();
    double ans(0);
    for(uint m=1; m<M; ++m)
      ans += subject_pri_->logp(mlm_->beta_subject(m));
    uint pch = mlm_->choice_nvars();
    if(pch>0) ans+= choice_pri_->logp(mlm_->beta_choice());
    return ans;
  }
  //------------------------------------------------------------
  // to be called by the constructor
  void DafeMlmBase::compute_xtx(){
    std::vector<Ptr<ChoiceData> > & d(mlm_->dat());
    uint psub = d[0]->subject_nvars();
    uint pch = d[0]->choice_nvars();

    xtx_subject_.resize(psub);
    xtx_subject_=0;

    xtx_choice_.resize(pch);
    if(pch>0) xtx_choice_=0;

    for(uint i=0; i<d.size(); ++i){
      Ptr<ChoiceData> dp = d[i];
      const Vec & xsub(dp->Xsubject());
      xtx_subject_.add_outer(xsub);
      if(pch>0){
	for(uint m = 0; m< mlm_->Nchoices(); ++m){
	  const Vec & xch(dp->Xchoice(m));
	  xtx_choice_.add_outer(xch); }}}}

  const Spd & DafeMlmBase::xtx_subject()const{return xtx_subject_;}
  const Spd & DafeMlmBase::xtx_choice()const{return xtx_choice_;}
  Ptr<MvnModel> DafeMlmBase::subject_pri()const{return subject_pri_;}
  Ptr<MvnModel> DafeMlmBase::choice_pri()const{return choice_pri_;}

  // ======================================================================
  // Target function for use with Metropolis Hastings samplers
  class LesSubjectTarget : public TargetFun{
  public:
    LesSubjectTarget(uint Which, Mat & bigU, MLM *mod)
      : which(Which),
	U(bigU),
	mlm_(mod)
    {}
    LesSubjectTarget * clone()const{return new LesSubjectTarget(*this);}
    double operator()(const Vec &b)const;
  private:
    uint which;
    Mat &U;
    MLM *mlm_;
  };
  double LesSubjectTarget::operator()(const Vec &b)const{
    const VectorView Uvec(U.col(which));
    uint n = Uvec.size();
    const std::vector<Ptr<ChoiceData> > & dat(mlm_->dat());
    double ans=0;
    for(uint i=0; i<n; ++i){
      double u = Uvec[i];
      Ptr<ChoiceData> d(dat[i]);
      double eta = b.affdot(d->Xsubject());
      eta+= mlm_->predict_choice(d, which);
      ans+= dexv(u, eta, 1, true);
    }
    return ans;
  }
  // ======================================================================
  class LesChoiceTarget : public TargetFun{
  public:
    LesChoiceTarget(Mat &bigU, MLM *mod)
      : U(bigU),
        mlm_(mod){}
    LesChoiceTarget * clone()const{return new LesChoiceTarget(*this);}
    double operator()(const Vec &b)const;
  private:
    Mat &U;
    MLM *mlm_;
  };
  double LesChoiceTarget::operator()(const Vec &b)const{
    const std::vector<Ptr<ChoiceData> > & dat(mlm_->dat());
    double n = dat.size();
    uint M = mlm_->Nchoices();
    double ans=0;
    for(uint i=0; i<n; ++i){
      Ptr<ChoiceData> d(dat[i]);
      for(uint m=0; m<M; ++m){
	double eta = mlm_->predict_subject(d,m);
	eta += b.affdot(d->Xchoice(m));
	double u = U(i,m);
	ans+= dexv(u, eta, 1, true);}}
    return ans;
  }
  // ======================================================================
  DafeMlm::DafeMlm(MultinomialLogitModel *mod,
		   Ptr<MvnModel> SubjectPri,
		   Ptr<MvnModel> ChoicePri,
		   double Tdf,
		   bool draw_b0)
    : DafeMlmBase(mod, SubjectPri, ChoicePri, draw_b0),
      mlm_(mod),
      mu(-0.577215664902),
      sigsq(1.64493406685),
      U(mod->dat().size(), mod->Nchoices())
  {
    uint M = mod->Nchoices();
    uint psub = mod->subject_nvars();
    const Spd & Ominv(subject_pri()->siginv());
    Ominv_mu_subject = Ominv * subject_pri()->mu();
    for(uint m=0; m<M; ++m){
      // just need to get the dimensions right, for now
      NEW(MvtIndepProposal, prop)(Ominv_mu_subject, Ominv, Tdf);
      subject_proposals_.push_back(prop);

      LesSubjectTarget target(m, U, mlm_);
      NEW(MH, sam)(target, prop);
      subject_samplers_.push_back(sam);

      Vec tmp(psub);
      xtu_subject.push_back(tmp);
    }

    uint pch = mod->choice_nvars();
    if(pch>0){
      LesChoiceTarget target(U, mlm_);
      choice_proposal_ = new MvtIndepProposal(choice_pri()->mu(),
                                              choice_pri()->siginv(),
                                              Tdf);
      choice_sampler_ = new MH(target, choice_proposal_);
      Ominv_mu_choice = choice_pri()->siginv() * choice_pri()->mu();
      xtu_choice = Vec(pch);
    }
  }

  // ======================================================================
  void DafeMlm::draw(){
    impute_latent_data();
    uint M = subject_samplers_.size();
    for(uint m= mlo(); m<M; ++m) draw_subject(m);
    if(mlm_->choice_nvars()>0) draw_choice();
  }

  // ======================================================================
  void DafeMlm::impute_latent_data(){

    std::vector<Ptr<ChoiceData> > & dat(mlm_->dat());
    uint n = dat.size();
    uint M = dat[0]->nchoices();

    U.resize(n,M);
    Vec eta(M);
    Vec u(M);
    Vec logz2(2);
    for(uint m=0; m<M; ++m) xtu_subject[m] =0;
    uint pch = mlm_->choice_nvars();
    if(pch>0) xtu_choice = 0;

    for(uint i=0; i<n; ++i){
      Ptr<ChoiceData> dp = dat[i];
      mlm_->fill_eta(*dp, eta);
      uint y = dp->value();
      double loglam = lse(eta);
      double logzmin = rlexp(loglam);
      logz2[0] = logzmin;
      u[y] = mu- logzmin;
      const Vec & xsub(dp->Xsubject());
      for(uint m=0; m<M; ++m){
	if(m!=y){
	  logz2[1] =rlexp(eta[m]);
	  double logz = lse(logz2);
	  u[m] = mu-logz;}
	xtu_subject[m].axpy(xsub, u[m]);
	if(pch>0){
	  const Vec &xch(dp->Xchoice(m));
	  xtu_choice.axpy(xch, u[m]);}
      } // m
      U.row(i) = u;
    }//i
  }// impute_latent_data

  // ======================================================================


  inline void Breg(Vec &b, Spd &ivar, double sigsq,
		   const Vec &xty, const Spd &xtx,
		   const Vec &Ominv_b, const Spd &Ominv){
    ivar = xtx/sigsq+Ominv;
    b = xty/sigsq + Ominv_b;
    b = ivar.solve(b);
  }
  // ======================================================================
  void DafeMlm::draw_subject(uint i){
    Vec b;
    Spd Ivar;
    const Spd & Ominv(subject_pri()->siginv());

    Breg(b, Ivar, sigsq, xtu_subject[i], xtx_subject(),
	 Ominv_mu_subject, Ominv);

    Ptr<MvtIndepProposal> prop = subject_proposals_[i];
    prop->set_ivar(Ivar);
    prop->set_mu(b);

    b=mlm_->beta_subject(i);
    b = subject_samplers_[i]->draw(b);
    mlm_->set_beta_subject(b,i);
  }
  // ======================================================================
  void DafeMlm::draw_choice(){
    Vec b;
    Spd Ivar;
    const Spd &Ominv(choice_pri()->siginv());
    Breg(b,Ivar, sigsq, xtu_choice, xtx_choice(),
	 Ominv_mu_choice, Ominv);
    choice_proposal_->set_mu(b);
    choice_proposal_->set_ivar(Ivar);

    b = choice_sampler_->draw(mlm_->beta_choice());
    if(b!=mlm_->beta_choice()){
      mlm_->set_beta_choice(b);
    }
  }

  //______________________________________________________________________


  class DafeLoglike{
  public:
    DafeLoglike(MLM *, uint m, bool choice=false);
    double operator()(const Vec &Beta)const;
    //    virtual DafeLoglike * clone()const;
  private:
    mutable MLM *mlm_;
    mutable Vec x;
    uint m;
    bool choice;
  };


  DafeLoglike::DafeLoglike(MLM *mod, uint which_choice, bool is_choice)
    : mlm_(mod),
      m(which_choice),
      choice(is_choice)
  {}

//   DafeLoglike * DafeLoglike::clone()const{
//     return new DafeLoglike(*this);}

  double DafeLoglike::operator()(const Vec &beta)const{
    double ans=0;
    if(choice){
      x = mlm_->beta_choice();
      mlm_->set_beta_choice(beta);
      ans = mlm_->loglike();
      mlm_->set_beta_choice(x);
    }else{
      x = mlm_->beta_subject(m);
      mlm_->set_beta_subject(beta,m);
      ans = mlm_->loglike();
      mlm_->set_beta_subject(x,m);
    }
    return ans;
  }

  struct Logp{
    Logp(boost::shared_ptr<DafeLoglike> L, Ptr<MvnModel> P)
      : loglike(L), pri(P){}
    double operator()(const Vec &x)const{
      return (*loglike)(x) + pri->logp(x);}
    boost::shared_ptr<DafeLoglike> loglike;
    Ptr<MvnModel> pri;
  };

  //______________________________________________________________________

  DafeRMlm::DafeRMlm(MultinomialLogitModel *mod,
			 Ptr<MvnModel> SubjectPri,
			 Ptr<MvnModel> ChoicePri,
			 double Tdf)
      : DafeMlmBase(mod, SubjectPri, ChoicePri),
        mlm_(mod)
  {
    uint M = mod->Nchoices();
    for(uint m=0; m<M; ++m){
      boost::shared_ptr<DafeLoglike> loglike(new DafeLoglike(mlm_,m));
      Logp logpost(loglike, subject_pri());
      Ptr<MH> sam = new MH(logpost,subject_proposals_[m]);
      subject_samplers_.push_back(sam);
    }

    uint pch = mlm_->choice_nvars();
    if(pch>0){
      boost::shared_ptr<DafeLoglike> choice_loglike(new DafeLoglike(mlm_,0, true));
      Logp choice_logpost(choice_loglike, choice_pri());
      choice_sampler_ = new MH(choice_logpost, choice_proposal_);
    }

  }
  // ======================================================================
  void DafeRMlm::draw(){
    // no need to draw beta 0
    for(uint m=mlo(); m<mlm_->Nchoices(); ++m) draw_subject(m);
    if(mlm_->choice_nvars()>0) draw_choice();
  }

  // ======================================================================
  void DafeRMlm::draw_subject(uint i){
    Vec b = mlm_->beta_subject(i);
    b = subject_samplers_[i]->draw(b);
    mlm_->set_beta_subject(b,i);
  }

  void DafeRMlm::draw_choice(){
    if(mlm_->choice_nvars()==0) return;
    Vec b = mlm_->beta_choice();
    b = choice_sampler_->draw(b);
    mlm_->set_beta_choice(b);
  }

// not sure why we need refresh_proposals?
//   void DafeRMlm::refresh_proposals(){
//     refresh_xtx();
//     uint M = mlm_->Nchoices();

//     for(uint m=0; m<M; ++m){
//       Spd Ivar = xtx_subject/sigsq + subject_pri->siginv();
//       subject_proposals[m]->set_ivar(Ivar);
//     }

//     uint pch = mlm_->choice_nvars();
//     if(pch>0){
//       Spd Ivar = xtx_choice/sigsq + choice_pri->siginv();
//       choice_proposal->set_ivar(Ivar);
//     }
//   }

}
