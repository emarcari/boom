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
#include <distributions.hpp>

namespace BOOM{

  typedef MvnBase MB;

  using LinAlg::Id;

  MvnSuf::MvnSuf(uint p)
    : sum_(p, 0.0),
      sumsq_(p, 0.0),
      n_(0.0),
      sym_(false)
  {}

  MvnSuf::MvnSuf(double n, const Vec &sum, const Spd & sumsq)
      : sum_(sum),
        sumsq_(sumsq),
        n_(n),
        sym_(false)
  {}

  MvnSuf::MvnSuf(const MvnSuf &rhs)
    : Sufstat(rhs),
      SufstatDetails<VectorData>(rhs),
      sum_(rhs.sum_),
      sumsq_(rhs.sumsq_),
      n_(rhs.n_),
      sym_(rhs.sym_)
  {}

  MvnSuf *MvnSuf::clone() const{ return new MvnSuf(*this);}

  void MvnSuf::clear(){
    sum_=0;
    sumsq_=0;
    n_=0;
    sym_ = false;
  }

  void MvnSuf::resize(uint p){
    sum_.resize(p);
    sumsq_.resize(p);
    clear();
  }

  void MvnSuf::update_raw(const Vec & x){
    n_+=1.0;
    sum_ += x;
    sumsq_.add_outer(x,1.0,false);
    sym_ = false;
  }

  void MvnSuf::Update(const VectorData &X){
    const Vec &x(X.value());
    update_raw(x);
  }

  void MvnSuf::add_mixture_data(const Vec &x, double prob){
    n_ += prob;
    sum_.axpy(x, prob);
    sumsq_.add_outer(x, prob, false);
    sym_ = false;
  }

  const Vec & MvnSuf::sum()const{return sum_;}
  const Spd & MvnSuf::sumsq()const{
    check_symmetry();
    return sumsq_;}
  double MvnSuf::n()const{return n_;}

  void MvnSuf::check_symmetry()const{
    if(!sym_){
      sumsq_.reflect();
      sym_ = true;
    }
  }

  Vec MvnSuf::ybar()const{
    if(n()>0) return sum_/n();
    return sum_.zero();}

  Spd MvnSuf::sample_var()const{
    if(n()>1) return center_sumsq()/(n()-1);
    return Id(sum_.size());
  }

  Spd MvnSuf::var_hat()const{
    if(n()>0) return center_sumsq()/n();
    return Id(sum_.size());
  }

  Spd MvnSuf::center_sumsq(const Vec &mu)const{
    double N = n();
    Spd ans = sumsq();
    ans.add_outer(mu, N);
    ans.add_outer2(mu, sum_, -1);
    return ans;
  }

  Spd MvnSuf::center_sumsq()const{
    return center_sumsq(ybar());}

  void MvnSuf::combine(Ptr<MvnSuf> s){
    sum_ += s->sum_;
    sumsq_ += s->sumsq_;
    n_ += s->n_;
    sym_ = sym_ && s->sym_;
  }

  void MvnSuf::combine(const MvnSuf & s){
    sum_ += s.sum_;
    sumsq_ += s.sumsq_;
    n_ += s.n_;
    sym_ = sym_ && s.sym_;
  }

  Vec MvnSuf::vectorize(bool minimal)const{
    Vec ans(sum_);
    ans.concat(sumsq_.vectorize(minimal));
    ans.push_back(n_);
    return ans;
  }

  Vec::const_iterator MvnSuf::unvectorize(Vec::const_iterator &v, bool){
    uint dim = sum_.size();
    sum_.assign(v, v+dim);
    v+=dim;
    sumsq_.unvectorize(v);
    n_ = *v; ++v;
    return v;
  }

  Vec::const_iterator MvnSuf::unvectorize(const Vec &v,
                                          bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  //======================================================================



  uint MB::dim()const{
    return mu().size();}


  double MB::Logp(const Vec &x, Vec &g, Mat &h, uint nd)const{
    double ans = dmvn(x,mu(), siginv(), ldsi(), true);
    if(nd>0){
      g = -(siginv() * (x-mu()));
      if(nd>1) h = -siginv();}
    return ans;}

  typedef MvnBaseWithParams MBP;

  MBP::MvnBaseWithParams(uint p, double mu, double sigsq)
    : ParamPolicy(new VectorParams(p,mu),
		  new SpdParams(p,sigsq))
  {}

    // N(mu,V)... if(ivar) then V is the inverse variance.
  MBP::MvnBaseWithParams(const Vec &mean, const Spd &V, bool ivar)
    : ParamPolicy(new VectorParams(mean), new SpdParams(V,ivar))
  {}


  MBP::MvnBaseWithParams(Ptr<VectorParams> mu, Ptr<SpdParams> S)
    : ParamPolicy(mu,S)
  {}

  MBP::MvnBaseWithParams(const MvnBaseWithParams &rhs)
    : Model(rhs),
      VectorModel(rhs),
      MvnBase(rhs),
      ParamPolicy(rhs),
      LocationScaleVectorModel(rhs)
  {}

  Ptr<VectorParams> MBP::Mu_prm(){
    return ParamPolicy::prm1();}
  const Ptr<VectorParams> MBP::Mu_prm()const{
    return ParamPolicy::prm1();}

  Ptr<SpdParams> MBP::Sigma_prm(){
    return ParamPolicy::prm2();}
  const Ptr<SpdParams> MBP::Sigma_prm()const{
    return ParamPolicy::prm2();}

  const Vec & MBP::mu()const{return Mu_prm()->value();}
  const Spd & MBP::Sigma()const{return Sigma_prm()->var();}
  const Spd & MBP::siginv()const{return Sigma_prm()->ivar();}
  double MBP::ldsi()const{return Sigma_prm()->ldsi();}

  void MBP::set_mu(const Vec &v){Mu_prm()->set(v);}
  void MBP::set_Sigma(const Spd &s){Sigma_prm()->set_var(s);}
  void MBP::set_siginv(const Spd &ivar){Sigma_prm()->set_ivar(ivar);}
  void MBP::set_S_Rchol(const Vec &sd, const Mat &L){
      Sigma_prm()->set_S_Rchol(sd,L); }


}
