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

#include <Models/BinomialModel.hpp>
#include <cassert>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

  typedef BinomialSuf BS;
  typedef BinomialModel BM;

  BS::BinomialSuf()
    : SufTraits(),
      sum_(0),
      nobs_(0)
  {}

  BS::BinomialSuf(const BS &rhs)
    : Sufstat(rhs),
      SufTraits(rhs),
      sum_(rhs.sum_),
      nobs_(rhs.nobs_)
  {}

  BS * BS::clone()const{return new BS(*this);}

  void BS::clear(){ nobs_ = sum_ = 0;}
  void BS::Update(const IntData &d){
    int y = d.value();
    sum_ += y;
    nobs_ += 1;
  }

  void BS::update_raw(double y){
    sum_ += y;
    nobs_ += 1;
  }

  double BS::sum()const{return sum_;}
  double BS::nobs()const{return nobs_;}

  void BS::combine(Ptr<BS> s){
    sum_ += s->sum_;
    nobs_ += s->nobs_;
  }
  void BS::combine(const BS & s){
    sum_ += s.sum_;
    nobs_ += s.nobs_;
  }

  Vec BS::vectorize(bool)const{
    Vec ans(2);
    ans[0] = sum_;
    ans[1] = nobs_;
    return ans;
  }

  Vec::const_iterator BS::unvectorize(Vec::const_iterator &v,
                                          bool){
    sum_ = *v;  ++v;
    nobs_ = *v; ++v;
    return v;
  }

  Vec::const_iterator BS::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  BM::BinomialModel(uint n, double p)
    : ParamPolicy(new UnivParams(p)),
      DataPolicy(new BS),
      PriorPolicy(),
      NumOptModel(),
      n_(n)
  {
    assert(n>0);
  }

  BM::BinomialModel(const BM & rhs)
    : Model(rhs),
      MLE_Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      NumOptModel(rhs),
      n_(rhs.n_)
  {}

  BM * BM::clone()const{return new BM(*this);}

  void BM::mle(){
    double p = suf()->sum()/(n_*suf()->nobs());
    set_prob(p);
  }

  uint BM::n()const{return n_;}
  double BM::prob()const{ return Prob_prm()->value();}
  void BM::set_prob(double p){ Prob_prm()->set(p);}

  Ptr<UnivParams> BM::Prob_prm(){ return ParamPolicy::prm();}
  const Ptr<UnivParams> BM::Prob_prm()const{ return ParamPolicy::prm();}

  double BM::Loglike(Vec &g, Mat &h, uint nd)const{
    double p = prob();
    double logp = log(p);
    double logp2 = log(1-p);

    double ntrials = n_ * suf()->nobs();
    double success = n_*suf()->sum();
    double fail = ntrials - success;

    double ans =  success * logp + fail * logp2;

    if(nd>0){
      double q = 1-p;
      g[0] = (success - p*ntrials)/(p*q);
      if(nd>1){
	h(0,0) = -1*(success/(p*p)  + fail/(q*q));
      }
    }
    return ans;
  }

  double BM::pdf(uint x,  bool logscale)const{
    if(x>n_)
      return logscale ? BOOM::infinity(-1) : 0;
    if(n_==1){
      double p = x==1 ? prob() : 1-prob();
      return logscale ? log(p) : p;
    }
    return dbinom(x,n_, prob(), logscale);
  }

  double BM::pdf(Ptr<Data> dp, bool logscale)const{
    return pdf(DAT(dp)->value(), logscale);}

  uint BM::sim()const{ return rbinom(n_, prob()); }
}