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
#include <Models/GammaModel.hpp>
#include <cmath>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

  typedef GammaSuf GS;
  typedef GammaModelBase GMB;

  GS::GammaSuf(){}
  GS::GammaSuf(const GammaSuf &rhs)
    : Sufstat(rhs),
      SufstatDetails<DataType>(rhs),
      sum_(rhs.sum_),
      sumlog_(rhs.sumlog_),
      n_(rhs.n_)
  {}

  GS *GS::clone() const{return new GS(*this);}

  void GS::clear(){sum_ = sumlog_ = n_=0;}
  void GS::Update(const DoubleData &dat){
    double x = dat.value();
    update_raw_data(x);
  }

  void GS::update_raw_data(double x){
    ++n_;
    sum_ += x;
    sumlog_ += log(x);
  }

  void GS::add_mixture_data(double y, double prob){
    n_ += prob;
    sum_ += prob * y;
    sumlog_ += prob * log(y);
  }

  double GS::sum()const{return sum_;}
  double GS::sumlog()const{return sumlog_;}
  double GS::n()const{return n_;}
  ostream & GS::display(ostream &out)const{
    out << "gamma::sum    = " << sum_ << endl
	<< "gamma::sumlog = " << sumlog_ <<  endl
	<< "gamma::n      = " << n_ << endl;
    return out;
  }


  void GS::combine(Ptr<GS> s){
    sum_ += s->sum_;
    sumlog_ += s->sumlog_;
    n_ += s->n_;
  }

  void GS::combine(const GS & s){
    sum_ += s.sum_;
    sumlog_ += s.sumlog_;
    n_ += s.n_;
  }

  Vec GS::vectorize(bool)const{
    Vec ans(3);
    ans[0] = sum_;
    ans[1] = sumlog_;
    ans[2] = n_;
    return ans;
  }

  Vec::const_iterator GS::unvectorize(Vec::const_iterator &v, bool){
    sum_ = *v;    ++v;
    sumlog_ = *v; ++v;
    n_ = *v;      ++v;
    return v;
  }

  Vec::const_iterator GS::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }


  //======================================================================
  GMB::GammaModelBase()
    : DataPolicy(new GammaSuf())
  {}

  GMB::GammaModelBase(const GMB &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      DataPolicy(rhs),
      DiffDoubleModel(rhs),
      NumOptModel(rhs),
      EmMixtureComponent(rhs)
  {}

  double GMB::Logp(double x, double &g, double &h, uint nd) const{
     double a = alpha();
     double b = beta();
     double ans = dgamma(x, a,b,true);
     if(nd>0) g = (a-1)/x-b;
     if(nd>1) h = -(a-1)/(x*x);
     return ans;
  }
  double GMB::sim() const{
    return rgamma(alpha(), beta());}

  void GMB::add_mixture_data(Ptr<Data> dp, double prob){
    double y = DAT(dp)->value();
    suf()->add_mixture_data(y, prob);
  }

  //======================================================================

  GammaModel::GammaModel(double a, double b, bool moments)
    : MLE_Model(),
      GMB(),
      ParamPolicy(new UnivParams(a), new UnivParams(b)),
      PriorPolicy()
  {
    if(moments){
      double mu =b;
      b = a/mu;
      set_beta(b);
    }
  }

  GammaModel::GammaModel(const GammaModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      GMB(rhs),
      ParamPolicy(rhs),
      PriorPolicy(rhs)
  {}

  GammaModel * GammaModel::clone()const{
    return new GammaModel(*this);}


  Ptr<UnivParams> GammaModel::Alpha_prm(){return ParamPolicy::prm1();}
  Ptr<UnivParams> GammaModel::Beta_prm(){return ParamPolicy::prm2();}
  const Ptr<UnivParams> GammaModel::Alpha_prm()const{return ParamPolicy::prm1();}
  const Ptr<UnivParams> GammaModel::Beta_prm()const{return ParamPolicy::prm2();}

  double GammaModel::alpha()const{return Alpha_prm()->value();}
  double GammaModel::beta()const{return Beta_prm()->value();}
  void GammaModel::set_alpha(double a){Alpha_prm()->set(a);}
  void GammaModel::set_beta(double b){Beta_prm()->set(b);}
  void GammaModel::set_params(double a, double b){
    set_alpha(a);
    set_beta(b);}

  inline double bad_gamma_loglike(double a,double b, Vec &g, Mat &h, uint nd){
    if(nd>0){
      g[0] = (a <=0) ? -(a+1) : 0;
      g[1] = (b <= 0) ? -(b+1) : 0;
      if(nd>1) h.set_diag(-1);
    }
    return BOOM::infinity(-1);
  }


  double GammaModel::Loglike(Vec &g, Mat &h, uint nd) const{
    double n = suf()->n();
    double sum =suf()->sum();
    double sumlog = suf()->sumlog();
    double a = alpha();
    double b = beta();
    if(a<=0 || b<=0) return bad_gamma_loglike(a, b,g,h,nd);

    double logb = log(b);
    double ans = n*(a*logb -lgamma(a))  + (a-1)*sumlog - b*sum;

    if(nd>0){
      g[0] = n*( logb -digamma(a) ) + sumlog;
      g[1] = n*a/b -sum;
      if(nd>1){
 	h(0,0) = -n*trigamma(a);
 	h(1,0) = h(0,1) = n/b;
 	h(1,1) = -n*a/(b*b);}}
    return ans;
  }

  void GammaModel::mle(){
    // can get good starting values;
    double n = suf()->n();
    double sum= suf()->sum();
    double sumlog = suf()->sumlog();

    double ybar = sum/n;        // arithmetic mean
    double gm = exp(sumlog/n);  // geometric mean
    double ss=0;
    for(uint i=0; i<dat().size(); ++i)
      ss+= pow(dat()[i]->value()-ybar, 2);
    double v = ss/(n-1);

    // method of moments estimates
    double b = ybar/v;
    double a = ybar*b;

    // one step newton refinement:
    // a = ybar *b;
    // b - exp(psi(ybar*b))/gm = 0
    double tmp = exp(digamma(ybar*b))/gm;
    double f =  b - tmp;
    double g = 1 - tmp*trigamma(ybar*b) * ybar;

    b-= f/g;
    a = b*ybar;
    set_params(a,b);
    NumOptModel::mle();
  }

  //======================================================================

}
