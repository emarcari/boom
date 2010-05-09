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

#include "ScaledChisqModel.hpp"
#include <cpputil/math_utils.hpp>
#include <cmath>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef ScaledChisqModel SCM;

  ScaledChisqModel::ScaledChisqModel(double nu)
    : GammaModelBase(),
      ParamPolicy(new UnivParams(nu)),
      PriorPolicy()
  {}

  ScaledChisqModel::ScaledChisqModel(const ScaledChisqModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      GammaModelBase(rhs),
      ParamPolicy(rhs),
      PriorPolicy(rhs)
  {}

  ScaledChisqModel * SCM::clone()const{return new SCM(*this);}

  Ptr<UnivParams> SCM::Nu_prm(){return ParamPolicy::prm();}
  const Ptr<UnivParams> SCM::Nu_prm()const{return ParamPolicy::prm();}

  const double & SCM::nu() const { return Nu_prm()->value();}
  void SCM::set_nu(double nu){Nu_prm()->set(nu);}

  // probability calculations
  double SCM::Loglike(Vec &g, Mat &h, uint nd) const {

    // loglike is a function of nu, derivatives are with respect to
    // nu.  however the model is w~Ga(nu/2, nu/2)

    double n = suf()->n();
    double sum =suf()->sum();
    double sumlog = suf()->sumlog();

    double nu = this->nu();
    if(nu <=0){
      double ans = infinity(-1);
      if(nd>0){
	g[0] = -nu;
	if(nd>1) h(0,0) = -1;
      }
      return ans;
    }
    double nu2 = nu/2.0;
    double lognu2 = log(nu2);
    double ans = n*(nu2*lognu2 - lgamma(nu2)) + (nu2-1)*sumlog - nu2*sum;
    if(nd>0){
      double halfn = n/2.0;
      g.front()=  halfn*(lognu2 + 1 - digamma(nu2)) + .5*(sum - sumlog);
      if(nd>1){
 	uint lo = 0;
 	h(lo,lo) = halfn*(1.0/nu - .5*trigamma(nu2));}}
    return ans;}

//   double SCM::pdf(dPtr dp, bool logscale) const{
//     return pdf(DAT(dp)->value(), logscale);}

//   double SCM::pdf(double x, bool logscale) const{
//     double nu_2 = nu()/2.0;
//     return dgamma(x, nu_2, nu_2, logscale); }

//   double SCM::Logp(const double x, double &g, double &h, uint nd) const{
//     double nu = this->nu();
//     double lognu = log(nu);
//     double ans = nu*lognu - lgamma(nu) + (nu-1)*log(x) - nu*x;
//     if(nd > 0){
//       g = (nu-1)/x - nu;
//       if(nd>1){
//  	h = -(nu-1)/(x*x);}}
//     return ans;}

//  double SCM::simdat() const{ double nu_2 = nu()/2.0; return rgamma(nu_2, nu_2);}

}
