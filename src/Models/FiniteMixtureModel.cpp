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

  FMM::FiniteMixtureModel(Ptr<LoglikeModel> mcomp, uint S)
    : MixtureDataPolicy(S),
      mixing_dist_(new MultinomialModel(S))
  {
    mixture_components_.reserve(S);
    for(uint s=0; s<S; ++s){
      Ptr<LoglikeModel> mod = mcomp->clone();
      mixture_components_.push_back(mod);
    }
    set_observers();
  }

  FMM::FiniteMixtureModel(Ptr<LoglikeModel> mcomp, Ptr<MultinomialModel> pi)
    : MixtureDataPolicy(pi->size()),
      mixing_dist_(pi)
  {
    uint S = pi->size();
    for(uint s=0; s<S; ++s){
      Ptr<LoglikeModel> mp = mcomp->clone();
      mixture_components_.push_back(mp);
    }
    set_observers();
  }

  FMM::FiniteMixtureModel(const FiniteMixtureModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      LoglikeModel(rhs),
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

  double FMM::loglike()const{
    const std::vector<Ptr<Data> >  &d(dat());
    uint n = d.size();
    uint S = size();

    logpi_ = log(pi());
    double ans = 0;
    wsp_.resize(S);

    for(uint i=0; i<n; ++i){
      for(uint s=0; s<S; ++s)
	wsp_[s] = logpi_[s] + mixture_components_[s]->pdf(d[i], true);
      ans += lse(wsp_);
    }
    return ans;
  }

  void FMM::mle(){
    ostringstream err;
    err << "mle is not yet implemented for finite mixture models." << endl;
    throw std::runtime_error(err.str());
  }

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

    wsp_.resize(S);
    set_logpi();

    const std::vector<Ptr<LoglikeModel> > &mod(mixture_components_);
    Ptr<MultinomialModel> mix(mixing_dist_);
    clear_component_data();
    for(uint i=0; i<n; ++i){
      dPtr dp = d[i];
      Ptr<CategoricalData> cd = hvec[i];
      for(uint s=0; s<S; ++s)
	wsp_[s] = logpi_[s] + mod[s]->pdf(dp, true);
      wsp_.normalize_logprob();
      uint h = rmulti(wsp_);
      cd->set(h);
      mod[h]->add_data(dp);
      mix->add_data(cd);
    }
  }

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
      wsp_[s] = logpi_[s] + mixture_components_[s]->pdf(dp, true);
    }
    if(logscale) return lse(wsp_);
    return sum(exp(wsp_));
  }

  uint FMM::size()const{return mixture_components_.size();}

  const Vec & FMM::pi()const{ return mixing_dist_->pi();}



  std::vector<Ptr<LoglikeModel> > FMM::mixture_components(){
    return mixture_components_;}
  const std::vector<Ptr<LoglikeModel> > FMM::mixture_components()const{
    return mixture_components_;}

  std::vector<Ptr<LoglikeModel> > FMM::models(){
    return mixture_components_;}
  const std::vector<Ptr<LoglikeModel> > FMM::models()const{
    return mixture_components_;}

  void FMM::observe_pi()const{ logpi_current_ = false; }

  double FMM::complete_data_loglike()const{
    double ans = mixing_dist_->loglike();
    uint S = size();
    for(uint s=0; s<S; ++s){
      ans+= mixture_components_[s]->loglike();
    }
    return ans;
  }

  double FMM::logpri()const{
    double ans = mixing_dist_->logpri();
    uint S = size();
    for(uint s =0; s<S; ++s) ans += mixture_components_[s]->logpri();
    return ans;
  }

  void FMM::sample_posterior(){
    progress();
    clear_component_data();
    impute_latent_data();
    mixing_dist_->sample_posterior();
    uint S = size();
    for(uint s=0; s<S; ++s){
      mixture_components_[s]->sample_posterior();
    }
  }

  void FMM::set_method(Ptr<PosteriorSampler>){
    string err;
    err += "set_method not defined for FiniteMixtureModel ";
    throw std::runtime_error(err);
  }
}
