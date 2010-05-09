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
#include "MultinomialModel.hpp"
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/DirichletModel.hpp>
#include <Models/PosteriorSamplers/MultinomialDirichletSampler.hpp>
#include <distributions.hpp>

#include <cmath>
#include <stdexcept>

namespace BOOM{
  typedef MultinomialSuf MS;
  typedef CategoricalData CD;
  MS::MultinomialSuf(const uint p)
    : counts(p, 0.0)
  {}

  MS::MultinomialSuf(const MultinomialSuf &rhs)
    : Sufstat(rhs),
      SufstatDetails<CD>(rhs),
      counts(rhs.counts)
  {}

  MS * MS::clone()const{return new MS(*this);}

  void MS::Update(const CD &d){
    uint i = d.value();
    while(i>=counts.size()) counts.push_back(0);  // counts grows when needed
    ++counts[i]; }

  void MS::add_mixture_data(uint y, double prob){
    counts[y]+=prob;
  }

  void MS::update_raw(uint k){ ++counts[k]; }

  void MS::clear(){counts=0.0;}

  const Vec &MS::n()const{return counts;}

  void MS::combine(Ptr<MS> s){ counts += s->counts; }
  void MS::combine(const MS & s){ counts += s.counts; }

  Vec MS::vectorize(bool)const{
    return counts;
  }

  Vec::const_iterator MS::unvectorize(Vec::const_iterator &v,
                                      bool){
    uint dim = counts.size();
    counts.assign(v, v+dim);
    v+=dim;
    return v;
  }

  Vec::const_iterator MS::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it,minimal);
  }

  //======================================================================

  typedef MultinomialModel MM;
  typedef std::vector<string> StringVec;

  MM::MultinomialModel(uint p)
    : ParamPolicy(new VectorParams(p, 1.0/p)),
      DataPolicy(new MS(p)),
      ConjPriorPolicy()
  {}

  uint  count_levels(const StringVec &sv){
    std::set<string> s;
    for(uint i=0; i<sv.size(); ++i) s.insert(sv[i]);
    return s.size();
  }

  MM::MultinomialModel(const Vec &probs)
    : ParamPolicy(new VectorParams(probs)),
      DataPolicy(new MS(probs.size())),
      ConjPriorPolicy()
  {}

  MM::MultinomialModel(const StringVec &names)
    : ParamPolicy(new VectorParams(1)),
      DataPolicy(new MS(1)),
      ConjPriorPolicy()
  {
    std::vector<Ptr<CD> >
      dvec(make_catdat_ptrs(names));

    uint nlev= dvec[0]->size();
    Vec probs(nlev, 1.0/nlev);
    set_pi(probs);

    set_data(dvec);
    mle();
  }

  MM::MultinomialModel(const MM &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      ConjPriorPolicy(rhs),
      LoglikeModel(rhs),
      EmMixtureComponent(rhs)
  {}

  MM * MM::clone()const{return new MM(*this);}

  Ptr<VectorParams> MM::Pi_prm(){
    return ParamPolicy::prm();}
  const Ptr<VectorParams> MM::Pi_prm()const{
    return ParamPolicy::prm();}

  uint MM::nlevels()const{return pi().size();}
  const double & MM::pi(int s) const{ return pi()[s];}
  const Vec & MM::pi()const{return Pi_prm()->value();}
  void MM::set_pi(const Vec &probs){ Pi_prm()->set(probs); }

  uint MM::size()const{return pi().size();}

  double MM::loglike()const{
    double ans(0.0);
    const Vec &n(suf()->n());
    const Vec &p(pi());
    for(uint i=0; i<size(); ++i)ans+= n[i]*log(p[i]);
    return ans;
  }

  void MM::mle(){
    const Vec &n(suf()->n());
    double tot = sum(n);
    if(tot==0){
      Vec probs(size(), 1.0/size());
      set_pi(probs);
      return;
    }
    set_pi(n/tot);
  }

  double MM::pdf(Ptr<Data> dp, bool logscale)const{
    uint i = DAT(dp)->value();
    if(i >=size()){
      string msg = "too large a value passed to MultinomialModel::pdf";
      throw std::runtime_error(msg);
    }
    double p = pi(i);
    return logscale ? log(p) : p;
  }


  uint MM::simdat()const{ return rmulti(pi()); }

  void MM::add_mixture_data(Ptr<Data> dp, double prob){
    uint i = DAT(dp)->value();
    suf()->add_mixture_data(i,prob);
  }

  void MM::set_conjugate_prior(const Vec &nu){
    NEW(MultinomialDirichletSampler, sam)(this, nu);
    this->set_conjugate_prior(sam);
  }

  void MM::set_conjugate_prior(Ptr<DirichletModel> dir){
    NEW(MultinomialDirichletSampler, sam)(this, dir);
    ConjPriorPolicy::set_conjugate_prior(sam);
  }

  void MM::set_conjugate_prior(Ptr<MultinomialDirichletSampler> sam){
    ConjPriorPolicy::set_conjugate_prior(sam);
  }
}
