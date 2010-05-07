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

#include <Models/Glm/MvnGivenX.hpp>
#include <Models/Glm/Glm.hpp>
#include <Models/Glm/RegressionModel.hpp>
#include <Models/Glm/LogisticRegressionModel.hpp>
#include <Models/Glm/MLogitBase.hpp>
#include <distributions.hpp>
#include <cpputil/nyi.hpp>

namespace BOOM{

  MvnGivenX::MvnGivenX(const Vec &Mu, double nobs, double diag)
    : ParamPolicy(new VectorParams(Mu), new UnivParams(nobs) ),
      diagonal_weight_(diag),
      Lambda_(Mu.length(), 0),
      ivar_(new SpdParams(Mu.length(), 0.0)),
      xtwx_(Mu.length(), 0.0),
      sumw_(0)
  {}

  MvnGivenX::MvnGivenX(Ptr<VectorParams> Mu, Ptr<UnivParams> nobs, double diag)
    : ParamPolicy(Mu, nobs),
      diagonal_weight_(diag),
      Lambda_(Mu->dim(), 0),
      ivar_(new SpdParams(Mu->dim(), 0.0)),
      xtwx_(Mu->dim(), 0.0),
      sumw_(0)
  {}

  MvnGivenX::MvnGivenX(Ptr<VectorParams> Mu,
		       Ptr<UnivParams> nobs,
		       const Vec & Lambda,
		       double diag)
    : ParamPolicy(Mu, nobs),
      diagonal_weight_(diag),
      Lambda_(Lambda),
      ivar_(new SpdParams(Mu->dim(), 0.0)),
      xtwx_(Mu->dim(), 0.0),
      sumw_(0)
  {
    assert(Lambda_.size()==Mu->dim());
  }

  MvnGivenX::MvnGivenX(const MvnGivenX &rhs)
    : Model(rhs),
      VectorModel(rhs),
      MvnBase(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      diagonal_weight_(rhs.diagonal_weight_),
      Lambda_(rhs.Lambda_),
      ivar_(rhs.ivar_->clone()),
      xtwx_(rhs.xtwx_),
      sumw_(rhs.sumw_)
  {}

  MvnGivenX * MvnGivenX::clone()const{return new MvnGivenX(*this);}

  void MvnGivenX::add_x(const Vec &x, double w){
    xtwx_.add_outer(x, w, false);
    sumw_ += w;
    current_ = false;
  }

  void MvnGivenX::clear_xtwx(){
    xtwx_ = 0;
    sumw_ =0;
    current_ = false;
  }

  const Spd & MvnGivenX::xtwx()const{ return xtwx_; }

  void MvnGivenX::initialize_params(){}

  const Ptr<VectorParams> MvnGivenX::Mu_prm()const{ return prm1(); }
  const Ptr<UnivParams> MvnGivenX::Kappa_prm()const{ return prm2(); }
  Ptr<VectorParams> MvnGivenX::Mu_prm(){ return prm1(); }
  Ptr<UnivParams> MvnGivenX::Kappa_prm(){ return prm2(); }
  double MvnGivenX::diagonal_weight()const{return diagonal_weight_;}
  uint MvnGivenX::dim()const{ return mu().size(); }
  const Vec & MvnGivenX::mu()const{ return Mu_prm()->value(); }
  double MvnGivenX::kappa()const{ return Kappa_prm()->value(); }

  const Spd & MvnGivenX::Sigma()const{
    if(!current_) set_ivar();
    return ivar_->var();
  }

  const Spd & MvnGivenX::siginv()const{
    if(!current_) set_ivar();
    return ivar_->ivar();
  }

  double MvnGivenX::ldsi()const{
    if(!current_) set_ivar();
    return ivar_->ldsi();
  }

  void MvnGivenX::set_ivar()const{
    Spd ivar = xtwx_;
    if(sumw_>0.0){
      ivar/=sumw_;
      double w= diagonal_weight_;

      if(w>= 1.0){
	ivar.set_diag(ivar.diag());
      }else if(w>0.0){
	ivar*= (1-w);
	ivar.diag()/=(1-w);
      }
    }
    else ivar *= 0.0;

    ivar.diag() += Lambda_;

    ivar_->set_ivar(ivar);
    current_ = true;
  }


  Vec MvnGivenX::sim()const{
    return rmvn(mu(), Sigma());
  }

  double MvnGivenX::pdf(Ptr<Data> dp, bool logscale)const{
    double ans = dmvn(DAT(dp)->value(), mu(), siginv(), ldsi(), true);
    return logscale ? ans : exp(ans);
  }

  double MvnGivenX::loglike()const{
    nyi("MvnGivenX::loglike");
    return 0;
    /////////////////////////////////////
  }

  //______________________________________________________________________
//   MvnGivenXReg::MvnGivenXReg(Ptr<RegressionModel> reg,
// 			       Ptr<VectorParams> Mu,
// 			       double prior_nobs, double diag_wgt)
//     : MvnGivenX(Mu, prior_nobs, diag_wgt),
//       reg_(reg)
//   {}

//   MvnGivenXReg::MvnGivenXReg(Ptr<RegressionModel> reg,
// 			     Ptr<VectorParams> Mu,
// 			     const Vec & Lambda_,
// 			     double prior_nobs,
// 			     double diag_wgt)
//     : MvnGivenX(Mu, Lambda, prior_nobs, diag_wgt),
//       reg_(reg)
//   {}

//   MvnGivenXReg::MvnGivenXReg(const MvnGivenXReg &rhs)
//     : Model(rhs),
//       MvnGivenX(rhs),
//       reg_(rhs.reg_)
//   {}

//   MvnGivenXReg * MvnGivenXReg::clone()const{
//     return new MvnGivenXReg(*this);}

//   void MvnGivenXReg::refresh_xtwx(){
//     clear_xtwx();
//     ////////////////////////
//   }


//   const Spd & MvnGivenXReg::Sigma()const{
//     double sigsq = reg_->sigsq();

//   }
//   const Spd & MvnGivenXReg::siginv()const{

//   }

//   double MvnGivenXReg::ldsi()const{
//     double sigsq = reg_->sigsq();
//     double ans = MvnGivenX::ldsi();
//     uint p = mu().size();
//     ans += p * log(sigsq);
//     return ans;
//   }



  //______________________________________________________________________

  MvnGivenXLogit::MvnGivenXLogit(Ptr<LogisticRegressionModel> mod,
				 Ptr<VectorParams> beta_prior_mean,
				 Ptr<UnivParams> prior_sample_size,
				 double diag_wgt)
    : MvnGivenX(beta_prior_mean, prior_sample_size, diag_wgt),
      mod_(mod)
  {}

  MvnGivenXLogit::MvnGivenXLogit(Ptr<LogisticRegressionModel> mod,
				 Ptr<VectorParams> beta_prior_mean,
				 Ptr<UnivParams> prior_sample_size,
				 const Vec & Lambda,
				 double diag_wgt)
    : MvnGivenX(beta_prior_mean, prior_sample_size, Lambda, diag_wgt),
      mod_(mod)
  {}

  MvnGivenXLogit::MvnGivenXLogit(const MvnGivenXLogit &rhs)
    : Model(rhs),
      VectorModel(rhs),
      MvnGivenX(rhs),
      mod_(rhs.mod_)
  {}

  MvnGivenXLogit * MvnGivenXLogit::clone()const{
    return new MvnGivenXLogit(*this);}


  void MvnGivenXLogit::refresh_xtwx(){
    clear_xtwx();
    const std::vector<Ptr<BinaryRegressionData> > &d(mod_->dat());
    uint n = d.size();
    for(uint i=0; i<n; ++i) add_x(d[i]->x());
  }


  //______________________________________________________________________


}
