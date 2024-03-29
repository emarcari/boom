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

#include <Models/FiniteMixtureModel.hpp>
#include <cpputil/lse.hpp>
#include <boost/bind.hpp>
#include <distributions.hpp>
#include <stdexcept>

namespace BOOM{


  typedef FiniteMixtureModel FMM;

  FMM::FiniteMixtureModel(Ptr<MixtureComponent> mcomp, uint S)
    : MixtureDataPolicy(S),
      mixing_dist_(new MultinomialModel(S))
  {
    mixture_components_.reserve(S);
    for(uint s=0; s<S; ++s){
      Ptr<MixtureComponent> mod = mcomp->clone();
      mixture_components_.push_back(mod);
    }
    set_observers();
  }

  FMM::FiniteMixtureModel(Ptr<MixtureComponent> mcomp, Ptr<MultinomialModel> pi)
    : MixtureDataPolicy(pi->size()),
      mixing_dist_(pi)
  {
    uint S = pi->size();
    for(uint s=0; s<S; ++s){
      Ptr<MixtureComponent> mp = mcomp->clone();
      mixture_components_.push_back(mp);
    }
    set_observers();
  }

  FMM::FiniteMixtureModel(const FiniteMixtureModel &rhs)
    : Model(rhs),
      LatentVariableModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      mixture_components_(rhs.mixture_components_),
      mixing_dist_(rhs.mixing_dist_->clone())
  {
    uint S = size();
    for(uint s =0; s<S; ++s)
      mixture_components_[s] = rhs.mixture_components_[s]->clone();
    set_observers();
  }

  FMM * FMM::clone()const{return new FMM(*this);}


  void FMM::clear_component_data(){
    mixing_dist_->clear_data();
    uint S = size();
    for(uint s=0; s<S; ++s) mixture_components_[s]->clear_data();
  }

  void FMM::impute_latent_data(){
    const std::vector<Ptr<Data> >  &d(dat());
    std::vector<Ptr<CategoricalData> > hvec(latent_data());

    uint n = d.size();
    uint S = size();
    class_membership_probabilities_.resize(n, S);

    wsp_.resize(S);
    set_logpi();
    last_loglike_ = 0;
    const std::vector<Ptr<MixtureComponent> > &mod(mixture_components_);
    Ptr<MultinomialModel> mix(mixing_dist_);
    clear_component_data();
    for(uint i=0; i<n; ++i){
      dPtr dp = d[i];
      Ptr<CategoricalData> cd = hvec[i];
      if(dp->missing()){
        wsp_ = logpi_;
      }else if(which_mixture_component(i) > 0){
        int source = which_mixture_component(i);
        last_loglike_ += mod[source]->pdf(dp.get(), true);
        class_membership_probabilities_.row(i) = 0;
        class_membership_probabilities_(i, source) = 1.0;
        cd->set(source);
        mix->add_data(cd);
        mod[source]->add_data(dp);
        continue;
      }else{
        for(uint s=0; s<S; ++s){
          wsp_[s] = logpi_[s] + mod[s]->pdf(dp.get(), true);
        }
      }
      last_loglike_ += lse(wsp_);
      wsp_.normalize_logprob();
      class_membership_probabilities_.row(i) = wsp_;
      uint h = rmulti(wsp_);
      cd->set(h);
      mod[h]->add_data(dp);
      mix->add_data(cd);
    }
  }

  void FMM::class_membership_probability(Ptr<Data> dp, Vec &ans)const{
    int S = number_of_mixture_components();
    ans.resize(S);
    const Vec & log_pi(logpi());
    for(int s = 0; s < S; ++s){
      ans[s] = log_pi[s] + mixture_component(s)->pdf(dp.get(), true);
    }
    ans.normalize_logprob();
  }

  double FMM::last_loglike()const{
    return last_loglike_;}

  void FMM::set_observers(){
    mixing_dist_->Pi_prm()->add_observer(boost::bind(&FMM::observe_pi, this));
    logpi_current_ = false;
    ParamPolicy::set_models(mixture_components_.begin(),
			    mixture_components_.end());
    ParamPolicy::add_model(mixing_dist_);
  }

  void FMM::set_logpi()const{
    if(!logpi_current_){
      logpi_ = log(pi());
      logpi_current_ = true;
    }
  }

  double FMM::pdf(dPtr dp, bool logscale)const{
    if(!logpi_current_) logpi_ = log(pi());
    uint S = size();
    wsp_.resize(S);
    for(uint s=0; s<S; ++s){
      wsp_[s] = logpi_[s] + mixture_components_[s]->pdf(dp.get(), true);
    }
    if(logscale) return lse(wsp_);
    return sum(exp(wsp_));
  }

  uint FMM::size()const{return mixture_components_.size();}
  uint FMM::number_of_mixture_components()const{
    return mixture_components_.size();}

  const Vec & FMM::pi()const{ return mixing_dist_->pi();}
  const Vec & FMM::logpi()const{
    set_logpi();
    return logpi_;}

  Ptr<MultinomialModel> FMM::mixing_distribution(){
    return mixing_dist_;
  }

  const MultinomialModel * FMM::mixing_distribution()const{
    return mixing_dist_.get();
  }


  Ptr<MixtureComponent> FMM::mixture_component(int s){
    return mixture_components_[s];
  }

  const MixtureComponent *FMM::mixture_component(int s)const{
    return mixture_components_[s].get();
  }

  const Mat & FMM::class_membership_probability()const{
    return class_membership_probabilities_;
  }

  Vec FMM::class_assignment()const{
    std::vector<Ptr<CategoricalData> > hvec(latent_data());
    int n = hvec.size();
    Vec ans(n);
    for(int i = 0; i < n; ++i) ans[i] = hvec[i]->value();
    return ans;
  }

  std::vector<Ptr<MixtureComponent> > FMM::models(){
    return mixture_components_;}
  const std::vector<Ptr<MixtureComponent> > FMM::models()const{
    return mixture_components_;}

  void FMM::observe_pi()const{ logpi_current_ = false; }

  //======================================================================

  EmFiniteMixtureModel::EmFiniteMixtureModel(
      Ptr<EmMixtureComponent> prototype_mixture_component,
      uint state_space_size)
    : FiniteMixtureModel(prototype_mixture_component, state_space_size)
  {
    populate_em_mixture_components();
  }

  EmFiniteMixtureModel::EmFiniteMixtureModel(
      Ptr<EmMixtureComponent> prototype_mixture_component,
      Ptr<MultinomialModel> mixing_distribution)
      : FiniteMixtureModel(prototype_mixture_component,
                           mixing_distribution)
  {
    populate_em_mixture_components();
  }

  EmFiniteMixtureModel::EmFiniteMixtureModel(
      const EmFiniteMixtureModel &rhs)
      : FiniteMixtureModel(rhs)
  {
    populate_em_mixture_components();
  }

  EmFiniteMixtureModel * EmFiniteMixtureModel::clone()const{
    return new EmFiniteMixtureModel(*this);}

  void EmFiniteMixtureModel::populate_em_mixture_components(){
    for(int s = 0; s < mixing_distribution()->size(); ++s){
      em_mixture_components_.push_back(
          mixture_component(s).dcast<EmMixtureComponent>());
    }
  }

  double EmFiniteMixtureModel::loglike()const{
    const std::vector<Ptr<Data> >  &d(dat());
    uint n = d.size();
    uint S = size();

    const Vec &log_pi(logpi());
    Vec wsp(S);
    double ans = 0;

    for(uint i=0; i<n; ++i){
      for(uint s=0; s<S; ++s){
	wsp[s] = log_pi[s] + mixture_component(s)->pdf(d[i].get(), true);
      }
      ans += lse(wsp);
    }
    return ans;
  }

  void EmFiniteMixtureModel::mle(){
    double eps = 1e-5;
    double old_loglike = EStep();
    double crit = 1+eps;
    while(crit < eps){
      MStep(false);
      double loglike = EStep();
      crit = loglike - old_loglike;
    }
  }

  double EmFiniteMixtureModel::EStep(){
    Vec &wsp(wsp_);
    wsp.resize(number_of_mixture_components());
    const std::vector<Ptr<Data> > &data(dat());
    double ans = 0;
    const Vec &log_pi(logpi());
    for(int i = 0; i < data.size(); ++i){
      for(int s = 0; s < number_of_mixture_components(); ++s){
        wsp[s] = log_pi[s] + mixture_component(s)->pdf(data[i].get(), true);
      }
      double total = lse(wsp);
      ans += total;
      double normalizing_constant = 0;
      for(int s = 0; s < number_of_mixture_components(); ++s){
        wsp[s] = exp(wsp[s] - total);
        normalizing_constant += wsp[s];
      }
      for(int s = 0; s < number_of_mixture_components(); ++s){
        em_mixture_components_[s]->add_mixture_data(
            data[i], wsp[s] / normalizing_constant);
      }
    }
    return ans;
  }

  void EmFiniteMixtureModel::MStep(bool posterior_mode){
    for(int s = 0; s < number_of_mixture_components(); ++s){
      if(posterior_mode) em_mixture_component(s)->find_posterior_mode();
      else em_mixture_component(s)->mle();
    }
  }

  Ptr<EmMixtureComponent>
  EmFiniteMixtureModel::em_mixture_component(int s){
    return em_mixture_components_[s];
  }

  const EmMixtureComponent *
  EmFiniteMixtureModel::em_mixture_component(int s)const{
    return em_mixture_components_[s].get();
  }

}
