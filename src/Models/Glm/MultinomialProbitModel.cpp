/*
  Copyright (C) 2006 Steven L. Scott

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
#include "MultinomialProbitModel.hpp"
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
#include <TargetFun/TargetFun.hpp>
#include <Samplers/SliceSampler.hpp>
#include <LinAlg/SWEEP.hpp>

#include <functional>

namespace BOOM{

  typedef MultinomialProbitModel MNP;


  MNP::MultinomialProbitModel(const Mat & beta_subject,
			      const Vec & beta_choice,
			      const Spd & Sig)
    : ParamPolicy(make_beta(beta_subject, beta_choice), new SpdParams(Sig)),
      DataPolicy(),
      PriorPolicy(),
      LatentVariableModel(),
      imp_method(Slice),
      nchoices_(Sig.nrow()),
      subject_xdim_(beta_subject.nrow()),
      choice_xdim_(beta_choice.size())
  {
    setup_suf();
  }

  // the function make_catdat_ptrs can make a ResponseVec out of a
  // vector of strings or uints
//   MNP::MultinomialProbitModel(ResponseVec responses,
// 			      const Mat &Xsubject_info,
// 			      const Arr3 &Xchoice_info)
//     : ParamPolicy(make_beta(responses, Xsubject_info
//   {}
  // dim(Xchoice_info) = [#obs, #choices, #choice x's]

//   MNP::MultinomialProbitModel(ResponseVec responses,    // no choice information
// 			      const Mat &Xsubject_info);

  MNP::MultinomialProbitModel(const std::vector<Ptr<ChoiceData> > &d)
    : ParamPolicy(make_beta(d), new SpdParams(d[0]->nchoices())),
      DataPolicy(),
      PriorPolicy(),
      LatentVariableModel(),
      imp_method(Slice),
      nchoices_(d[0]->nchoices()),
      subject_xdim_(d[0]->subject_nvars()),
      choice_xdim_(d[0]->choice_nvars())
  {
    for(uint i=0; i<d.size(); ++i) add_data(d[i]);
    setup_suf();
  }

  MNP::MultinomialProbitModel(const MNP &rhs)
    : Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      LatentVariableModel(rhs),
      imp_method(rhs.imp_method),
      U(rhs.U),
      nchoices_(rhs.nchoices_),
      subject_xdim_(rhs.subject_xdim_),
      choice_xdim_(rhs.choice_xdim_),
      yyt_(rhs.yyt_),
      xtx_(rhs.xtx_),
      xty_(rhs.xty_)
  {}

  MNP * MNP::clone()const{return new MNP(*this);}
  void MNP::initialize_params(){}
  //------------------------------------------------------------
  class TrunMvnTF : public TargetFun{
  public:
    TrunMvnTF(const Spd & siginv)
      : mu(siginv.nrow()),
	Ivar(siginv),
	ldsi(siginv.logdet()),
	y(0)
    {}

    TrunMvnTF * clone()const{return new TrunMvnTF(*this); }
    void set_mu(const Vec &m){ mu = m;}
    void set_Ivar(const Spd & ivar){
      Ivar = ivar;
      ldsi = Ivar.logdet();
    }
    void set_y(uint Y){y = Y;}
    double operator()(const Vec &x)const{
      if(x.imax()!=y) return BOOM::negative_infinity();
      return dmvn(x,mu,Ivar, ldsi, true);
    }
  private:
    Vec mu;
    Spd Ivar;
    double ldsi;
    uint y;
  };

  //------------------------------------------------------------

  void MNP::impute_latent_data(){
    DatasetType &d(dat());
    uint n = d.size();
    yyt_ = 0;
    xtx_ = 0;
    xty_ = 0;
    TrunMvnTF target(this->siginv());
    wsp.resize(Nchoices());
    for(uint i=0; i<n; ++i){
      Ptr<ChoiceData> dp = d[i];
      Vec &u(U[i]);
      impute_u(u, dp, target);
      update_suf(u, dp);
    }
  }

  //======================================================================
  double MNP::complete_data_loglike()const{
    const double log2pi = 1.83787706641;
    const Vec &beta(this->beta());
    double n = dat().size();
    double ans = -.5*n*log2pi + .5*n*Sigma_prm()->ldsi();
    double tmp = yty() + xtx_.Mdist(beta) - 2* beta.dot(xty_);
    ans -= .5*tmp;
    return ans;
  }
  //============================================================
  double MNP::pdf(Ptr<Data> dp, bool logsc)const{ return pdf(DAT(dp), logsc);}
  double MNP::pdf(Ptr<ChoiceData>, bool )const{
    throw_exception<std::runtime_error>("MultinomialProbit::pdf has not been defined");
    return 0.0;
  }

  //============================================================
  const Vec & MNP::beta()const{ return Beta_prm()->Beta();}

  Vec MNP::beta_subject(uint choice)const{
    const Vec &b(beta());
    uint p = subject_nvars();
    Vec::const_iterator cb = b.begin()+choice*p;
    Vec::const_iterator ce = cb+p;
    return Vec(cb,ce);
  }

  Vec MNP::beta_choice()const{
    const Vec &b(beta());
    uint p = choice_nvars();
    return Vec(b.end()-p, b.end());
  }

  const Spd & MNP::Sigma()const{ return Sigma_prm()->var();}
  const Spd & MNP::siginv()const{return Sigma_prm()->ivar();}
  double MNP::ldsi()const{return Sigma_prm()->ldsi();}

  Vec MNP::eta(Ptr<ChoiceData> dp)const{
    wsp.resize(dp->nchoices());
    eta(dp, wsp);
    return wsp;
  }

  Vec & MNP::eta(Ptr<ChoiceData> dp, Vec & ans)const{
    const Mat&X(dp->X());
    Ptr<GlmCoefs> b(Beta_prm());
    uint M = dp->nchoices();
    ans.resize(M);
    for(uint m=0; m<M; ++m){
      ConstVectorView x(X.row(m));
      ans[m] = b->predict(x);
    }
    return ans;
  }

  uint MNP::n()const{return dat().size();}
  uint MNP::xdim()const{
    return Nchoices()*subject_nvars() + choice_nvars();
  }

  uint MNP::subject_nvars()const{return subject_xdim_;}
  uint MNP::choice_nvars()const{return choice_xdim_;}
  uint MNP::Nchoices()const{return nchoices_;}

  void MNP::set_beta(const Vec &b){ Beta_prm()->set_beta(b);}
  void MNP::set_Sigma(const Spd &S){Sigma_prm()->set_var(S);}
  void MNP::set_siginv(const Spd &ivar){Sigma_prm()->set_ivar(ivar);}

  const Spd & MNP::xtx()const{return xtx_;}
  double MNP::yty()const{return traceAB(siginv(), yyt_);}
  const Spd & MNP::yyt()const{return yyt_;}
  const Vec & MNP::xty()const{return xty_;}

  void MNP::add_data(Ptr<ChoiceData> dp){
    Vec tmpu(dp->nchoices());
    tmpu.randomize();
    double mx = tmpu.max();
    uint y = dp->value();
    tmpu[y] = mx+1;
    U.push_back(tmpu);
    DataPolicy::add_data(dp);
  }

  void MNP::add_data(Ptr<Data> dp){
    add_data(DAT(dp));
  }

  Ptr<GlmCoefs> MNP::make_beta(const Mat & beta_subject, const Vec & beta_choice){
    nchoices_ = beta_subject.ncol();
    subject_xdim_ = beta_subject.nrow();
    choice_xdim_ = beta_choice.size();

    Vec beta;
    beta.reserve(beta_subject.size() + beta_choice.size());
    std::copy(beta_subject.begin(), beta_subject.end(), back_inserter(beta));
    std::copy(beta_choice.begin(), beta_choice.end(), back_inserter(beta));

    return new GlmCoefs(beta);
  }

  Ptr<GlmCoefs> MNP::make_beta(const std::vector<Ptr<ChoiceData> > &dv){
    Ptr<ChoiceData> dp = dv[0];
    uint nc = nchoices_ = dp->nchoices();
    uint psub = subject_xdim_ = dp->subject_nvars();
    uint pch = choice_xdim_ = dp->choice_nvars();

    uint p = nc*psub + pch;
    return new GlmCoefs(p, true);
//     Vec beta(nc * psub + pch, 0.0);
//     return new GlmCoefs(beta);
  }

  void MNP::setup_suf(){
    yyt_ = Spd(Nchoices(), 0.0);
    xtx_ = Spd(xdim());
    xty_ = Vec(xdim());
  }

  void MNP::impute_u(Vec &u, Ptr<ChoiceData> dp, TrunMvnTF & target){
    if(imp_method==Slice) impute_u_slice(u, dp, target);
    else if(imp_method==Gibbs) impute_u_Gibbs(u, dp, target);
    else throw_exception<std::runtime_error>("unrecognized method in impute_u");
  }

  void MNP::impute_u_slice(Vec &u, Ptr<ChoiceData> dp, TrunMvnTF & target){
    // slice sampler
    eta(dp, wsp);
    target.set_mu(wsp);
    uint y = dp->value();
    target.set_y(y);
    SliceSampler sam(target, true);
    u = sam.draw(u);
  }


  inline void rsw_mv(double &m, double &v, Vec &b, const Vec &u,
		     const Vec &mu, const Spd &siginv, uint pos){
    // compute the mean and variance of one dimension of a MVN(mu,
    // siginv^{-1}) conditional on the other dimension.
    v = 1.0/siginv(pos,pos);  // residual variance
    b = siginv.col(pos);
    b/= (-1.0/v);
    b[pos] = 0.0;
    m = mu[pos] + b.dot(u-mu);
  }

  void MNP::impute_u_Gibbs(Vec &u, Ptr<ChoiceData> dp, TrunMvnTF & ){
    uint y = dp->value();
    wsp = u;
    //    std::nth_element(wsp.begin(), wsp.begin()+1, wsp.end(), _1 > _2 );
    std::nth_element(wsp.begin(), wsp.begin()+1, wsp.end(),
                     std::greater<double>()  );
    double second_largest = wsp[1];
    eta(dp, wsp);

    const Spd &siginv(this->siginv());
    Vec b;
    double mean,v;
    rsw_mv(mean,v,b,u,wsp, siginv, y);
    u[y] = rtrun_norm(mean, sqrt(v), second_largest, true);
    for(uint i=0; i<dp->nchoices(); ++i){
      if(i!=y){
	rsw_mv(mean,v,b,u,wsp,siginv,i);
	u[i] = rtrun_norm(mean, sqrt(v), u[y], false);
      }
    }
  }


  void MNP::update_suf(const Vec &u, Ptr<ChoiceData> dp){
    const Spd & siginv(this->siginv());
    const Mat & X(dp->X());

    yyt_.add_outer(u);
    xtx_ += sandwich(X.t(), siginv);  // sum of XT siginv X
    xty_ += X.Tmult(siginv*u);        //
  }



}
