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

#include <Models/HMM/HMM2.hpp>
#include <Models/HMM/HmmFilter.hpp>
#include <Models/HMM/HmmDataImputer.hpp>

#include <Models/MarkovModel.hpp>
#include <Models/EmMixtureComponent.hpp>

#include <cpputil/math_utils.hpp>
#include <cpputil/string_utils.hpp>

#include <distributions.hpp>
#include <LinAlg/Types.hpp>
#include <stdexcept>
#include <cmath>


#include <boost/thread.hpp>
#include <boost/ref.hpp>

namespace BOOM{
typedef HiddenMarkovModel HMM;

//======================================================================

HMM::HiddenMarkovModel(std::vector<Ptr<Model> > Mix,
                       Ptr<MarkovModel> Mark )
    : mark_(Mark),
      mix_(Mix),
      filter_(new HmmFilter(mix_, mark_))
{
  ParamPolicy::set_models(mix_.begin(), mix_.end());
  ParamPolicy::add_model(mark_);
}

HMM::HiddenMarkovModel(const HMM &rhs)
    : Model(rhs),
      DataInfoPolicy(rhs),
      MLE_Model(rhs),
      DataPolicy(rhs),
      ParamPolicy(),
      PriorPolicy(rhs),
      LoglikeModel(rhs),
      mark_(rhs.mark_->clone()),
      mix_(rhs.state_space_size())
{
  for(uint i=0; i<state_space_size(); ++i)
    mix_[i] =rhs.mix_[i]->clone();

  NEW(HmmFilter, f)(mix_,mark_);
  set_filter(f);
}

HMM * HMM::clone()const{return new HMM(*this);}

void HMM::randomly_assign_data(){
  clear_client_data();
  uint S = state_space_size();
  Vec prob(S, 1.0/S);
  for(uint s=0; s<nseries(); ++s){
    const DataSeriesType & ts(dat(s));
    uint n = ts.size();
    for(uint i=0; i<n; ++i){
      uint h = rmulti(prob);
      mix_[h]->add_data(ts[i]);}}
}

void HMM::initialize_params(){
  randomly_assign_data();
  uint S = state_space_size();
  Mat Q(S,S, 1.0/S);
  set_Q(Q);
  for(uint s=0; s<S; ++s) mix_[s]->sample_posterior();
}


const Vec & HMM::pi0()const{return mark_->pi0();}
const Mat & HMM::Q()const{return mark_->Q();}

void HMM::set_pi0(const Vec &pi0){ mark_->set_pi0(pi0);}
void HMM::set_Q(const Mat &Q){mark_->set_Q(Q);}
void HMM::set_filter(Ptr<HmmFilter> f){filter_ = f;}

void HMM::fix_pi0(const Vec &Pi0){ mark_->fix_pi0(Pi0); }
void HMM::fix_pi0_stationary(){mark_->fix_pi0_stationary();}
void HMM::fix_pi0_uniform(){mark_->fix_pi0_uniform();}
void HMM::free_pi0(){mark_->free_pi0();}
bool HMM::pi0_fixed()const{return mark_->pi0_fixed();}

uint HMM::io_params(IO io_prm){
  uint ans = Model::io_params(io_prm);
  if(!!loglike_) loglike_->io(io_prm);
  if(!!logpost_) logpost_->io(io_prm);
  return ans;
}

uint HMM::state_space_size()const{return mix_.size();}

double HMM::pdf(dPtr dp, bool logscale) const{
  Ptr<DataSeriesType> dat = DAT(dp);
  double ans =  filter_->loglike(*dat);
  return logscale ? ans : exp(ans);
}

void HMM::clear_client_data(){
  mark_ -> clear_data();
  uint S = state_space_size();
  for(uint s=0; s<S; ++s) mix_[s]->clear_data();
}

void HMM::clear_prob_hist(){
  for(std::map<Ptr<Data>, Vec >::iterator it = prob_hist_.begin();
      it!=prob_hist_.end(); ++it){
    it->second = 0.0;}}

std::vector<Ptr<Model> > HMM::mixture_components(){return mix_;}
Ptr<Model> HMM::mixture_component(uint s){return mix_[s];}

Ptr<MarkovModel> HMM::mark(){return mark_;}

double HMM::loglike()const{
  uint ns = nseries();
  double ans=0;
  for(uint series=0; series < ns; ++series){
    const DataSeriesType & ts(dat(series));
    ans += filter_->loglike(ts);
  }
  return ans;
}

//======================================================================

HMM_EM::HMM_EM(std::vector<Ptr<EmMixtureComponent> >Mix,
               Ptr<MarkovModel> Mark)
    : HiddenMarkovModel(tomod(Mix), Mark),
      mix_(Mix),
      eps(1e-5)
{
  set_filter(new HmmEmFilter(mix_, mark()));
}



std::vector<Ptr<Model> >
HMM_EM::tomod(const std::vector<Ptr<EMC> > &Mix)const{
  std::vector<Ptr<Model> > ans(Mix.begin(), Mix.end());
  return ans;
}

HMM_EM::HMM_EM(const HMM_EM & rhs)
    : Model(rhs),
      DataInfoPolicy(rhs),
      MLE_Model(rhs),
      HiddenMarkovModel(rhs),
      mix_(rhs.mix_),
      eps(rhs.eps)
{
  for(uint i=0; i<mix_.size(); ++i) mix_[i] = rhs.mix_[i]->clone();
  set_mixture_components(mix_.begin(), mix_.end());
  set_filter(new HmmEmFilter(mix_, mark()));
}

HMM_EM * HMM_EM::clone()const{ return new HMM_EM(*this);}

void HMM_EM::find_mode(bool bayes, bool save_history){
  double oldloglike = Estep(bayes);
  double loglike = oldloglike;
  double crit = eps + 1;
  while(crit >eps){
    progress();
    Mstep(bayes);
    loglike = Estep(bayes);
    crit = loglike - oldloglike;
    oldloglike = loglike;
    if(save_history){
      io_params(WRITE);
    }
  }
}

void HMM_EM::mle(){ find_mode(false, trace_); }

void HMM_EM::find_posterior_mode(){ find_mode(true, trace_); }

void HMM::save_loglike(const string & fname, uint ping){
  if(!loglike_) loglike_ = new UnivParams(0.0);
  loglike_->set_fname(fname);
  loglike_->set_bufsize(ping);
}

void HMM::save_logpost(const string & fname, uint ping){
  if(!logpost_) logpost_ = new UnivParams(0.0);
  logpost_->set_fname(fname);
  logpost_->set_bufsize(ping);
}

void HMM::set_loglike(double ell){
  if(!loglike_){
    ostringstream err;
    err << "HMM:  You need to call 'save_loglike' "
        << "(and specify the file where you want it saved) "
        << " before you call 'set_loglike'."
        << endl;
    throw std::runtime_error(err.str());
  }
  loglike_->set(ell);
}

void HMM::write_loglike(double ell){
  set_loglike(ell);
  loglike_->io(WRITE);
}

void HMM::set_logpost(double ell){
  if(!logpost_){
    ostringstream err;
    err << "HMM:  You need to call 'save_logpost' "
        << "(and specify the file where you want it saved) "
        << " before you call 'set_logpost'."
        << endl;
    throw std::runtime_error(err.str());
  }
  logpost_->set(ell);
}

void HMM::write_logpost(double ell){
  set_logpost(ell);
  logpost_->io(WRITE);
}

void HMM_EM::trace(bool t){ trace_ = t; }

void HMM_EM::set_epsilon(double Eps){eps =Eps;}

void HMM_EM::initialize_params(){
  randomly_assign_data();
  uint S = state_space_size();
  for(uint h=0; h<S; ++h) mix_[h]->mle();
  Mat Q(S,S,1.0/S);
  set_Q(Q);
}

double HMM::impute_latent_data(){
  if(nthreads()>0)
    return impute_latent_data_with_threads();

  clear_client_data();
  double ans=0;
  uint ns = nseries();
  for(uint series = 0; series<ns; ++series){
    const DataSeriesType & ts(dat(series));
    ans += filter_->fwd(ts);
    filter_->bkwd_sampling(ts);}
  if(!!loglike_) set_loglike(ans);
  if(!!logpost_) set_logpost(ans + logpri());
  return ans;
}


double HMM_EM::Estep(bool bayes){
  clear_client_data();
  double ans=0;
  uint ns = nseries();
  for(uint series = 0; series<ns; ++series){
    const DataSeriesType & ts(dat(series));
    ans += filter_->fwd(ts);
    filter_->bkwd_smoothing(ts);}
  if(bayes){
    ans += mark()->logpri();
    for(uint s=0; s<state_space_size(); ++s) ans += mix_[s]->logpri();
  }
  return ans;
}

void HMM_EM::Mstep(bool bayes){
  uint S = mix_.size();
  for(uint s=0; s<S; ++s){
    if(bayes) mix_[s]->find_posterior_mode();
    else mix_[s]->mle();
  }
  if(bayes) mark()->find_posterior_mode();
  else mark()->mle();
}

////////////////////////////////////////////////////////////////////////////

void HMM::set_nthreads(uint n){
  workers_.clear();
  for(uint i=0; i<n; ++i){
    NEW(HmmDataImputer, imp)(this, i, n);
    workers_.push_back(imp);}}

uint HMM::nthreads()const{ return workers_.size();}

double HMM::impute_latent_data_with_threads(){
  clear_client_data();

  boost::thread_group tg;
  for(uint i = 0; i<nthreads(); ++i){
    workers_[i]->setup(this);
    tg.add_thread(new boost::thread(boost::ref(*workers_[i])));
  }
  tg.join_all();
  uint S = state_space_size();
  double loglike=0;
  for(uint i=0; i<nthreads(); ++i){
    loglike += workers_[i]->loglike();
    mark_->combine_data(*workers_[i]->mark(), true);
    for(uint s=0; s<S; ++s) mix_[s]->combine_data(*workers_[i]->models(s), true);
  }
  return loglike;
}


} // ends namespace BOOM
