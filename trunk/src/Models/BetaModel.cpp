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
#include "BetaModel.hpp"
#include <cmath>
#include <distributions.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  typedef BetaSuf BS;
  typedef BetaModel BM;

  BS * BS::clone()const{return new BS(*this);}
  void BS::Update(const DoubleData &d){
    double p = d.value();
    update_raw(p);
  }
  void BetaSuf::update_raw(double p){
    ++n_;
    sumlog_ += log(p);
    sumlogc_ += log(1-p);
  }

  BetaSuf * BS::abstract_combine(Sufstat *s){
      return abstract_combine_impl(this, s);}

  void BS::combine(Ptr<BS> s){
    n_ += s->n_;
    sumlog_ += s->sumlog_;
    sumlogc_ += s->sumlogc_;
  }

  void BS::combine(const BS & s){
    n_ += s.n_;
    sumlog_ += s.sumlog_;
    sumlogc_ += s.sumlogc_;
  }

  Vec BS::vectorize(bool)const{
    Vec ans(3);
    ans[0] = n_;
    ans[1] = sumlog_;
    ans[2] = sumlogc_;
    return ans;
  }

  Vec::const_iterator BS::unvectorize(Vec::const_iterator &v,
                                      bool){
    n_ = *v; ++v;
    sumlog_ = *v; ++v;
    sumlogc_ = *v; ++v;
    return v;
  }

  Vec::const_iterator BS::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & BS::print(ostream &out)const{
    out << n_ << " " << sumlog_ << " " << sumlogc_;
    return out;
  }

  BM::BetaModel()
    : ParamPolicy(new UnivParams(1.0), new UnivParams(1.0)),
      DataPolicy(new BS() ),
      PriorPolicy()
  {}

  BM::BetaModel(double a, double b)
    : ParamPolicy(new UnivParams(a), new UnivParams(b)),
      DataPolicy(new BS),
      PriorPolicy()
  {}

  BM::BetaModel(const BM &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      NumOptModel(rhs),
      DiffDoubleModel(rhs)
  {}

  BM * BM::clone()const{return new BM(*this);}

  Ptr<UnivParams> BM::Alpha(){return ParamPolicy::prm1();}
  Ptr<UnivParams> BM::Beta(){return ParamPolicy::prm2();}
  const Ptr<UnivParams> BM::Alpha()const{return ParamPolicy::prm1();}
  const Ptr<UnivParams> BM::Beta()const{return ParamPolicy::prm2();}

  const double &BM::a()const{return Alpha()->value();}
  const double &BM::b()const{return Beta()->value();}

  void BM::set_a(double alpha){Alpha()->set(alpha);}
  void BM::set_b(double beta){Beta()->set(beta);}
  void BM::set_params(double a, double b){set_a(a); set_b(b);}

  double BM::Loglike(Vec &g, Mat &h, uint nd) const{
    double alpha = a();
    double beta = b();

    double n = suf()->n();
    double sumlog = suf()->sumlog();
    double sumlogc = suf()->sumlogc();

    double ans = n*(lgamma(alpha + beta) - lgamma(alpha)-lgamma(beta));
    ans += (alpha-1)*sumlog + (beta-1)*sumlogc;

    if(nd>0){
      double psisum = digamma(alpha + beta);
      g[0] = n*(psisum-digamma(alpha)) + sumlog;
      g[1] = n*(psisum-digamma(beta)) + sumlogc;

      if(nd>1){
 	double trisum = trigamma(alpha+beta);
 	h(0,0) = n*(trisum - trigamma(alpha));
 	h(0,1) = h(1,0) = n*trisum;
 	h(1,1) = n*(trisum - trigamma(beta));}}
    return ans;
  }

  double BM::Logp(double x, double &d1, double &d2, uint nd) const{
    if(x<0 || x>1) return BOOM::infinity(-1);
    double inf = BOOM::infinity(1);
    double a = this->a();
    double b = this->b();
    if(a==inf || b==inf) return Logp_degenerate(x,d1,d2,nd);

    double ans = dbeta(x,a,b, true);

    double A = a-1;
    double B = b-1;
    double y = 1-x;

    if(nd>0){
      d1 = A/x - B/(y);
      if(nd>1) d2 = -A/(x*x) - B/(y*y);
    }
    return ans;
  }

  double BM::Logp_degenerate(double x, double &d1, double &d2, uint nd)const{
    double inf = BOOM::infinity(1);
    double a_inf = a()==inf;
    double b_inf = b()==inf;
    if(a_inf && b_inf) {
      report_error("both a and b are finite in BetaModel::Logp");
    }
    if(nd>0){
      d1=0;
      if(nd>1) d2=0;}
    if(b_inf) x= 1-x;
    return x==1.0 ?  0.0 : BOOM::infinity(-1);
  }

  double BM::sim() const { return rbeta(a(),b());}

  double beta_log_likelihood(double a, double b, const BetaSuf &suf){
    double n = suf.n();
    double sumlog = suf.sumlog();
    double sumlogc = suf.sumlogc();

    double ans = n*(lgamma(a + b) - lgamma(a)-lgamma(b));
    ans += (a-1)*sumlog + (b-1)*sumlogc;
    return ans;
  }
}
