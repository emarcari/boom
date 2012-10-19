/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <distributions.hpp>
#include <Models/ZeroInflatedLognormalModel.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>
#include <sstream>
#include <cpputil/report_error.hpp>
#include <Models/PosteriorSamplers/ZeroInflatedLognormalPosteriorSampler.hpp>
#include <boost/bind.hpp>

namespace BOOM{
  ZeroInflatedLognormalModel::ZeroInflatedLognormalModel()
      : gaussian_(new GaussianModel),
        binomial_(new BinomialModel),
        precision_(1e-8),
        log_probabilities_are_current_(false)
  {
    ParamPolicy::add_model(gaussian_);
    ParamPolicy::add_model(binomial_);
    binomial_->Prob_prm()->add_observer(create_binomial_observer());
  }

  ZeroInflatedLognormalModel::ZeroInflatedLognormalModel(
      const ZeroInflatedLognormalModel &rhs)
      : ParamPolicy(rhs),
        PriorPolicy(rhs),
        DoubleModel(rhs),
        EmMixtureComponent(rhs),
        gaussian_(rhs.gaussian_->clone()),
        binomial_(rhs.binomial_->clone()),
        precision_(rhs.precision_)
  {
    ParamPolicy::add_model(gaussian_);
    ParamPolicy::add_model(binomial_);
    binomial_->Prob_prm()->add_observer(create_binomial_observer());
  }

  ZeroInflatedLognormalModel * ZeroInflatedLognormalModel::clone()const{
    return new ZeroInflatedLognormalModel(*this);}

  double ZeroInflatedLognormalModel::pdf(Ptr<Data> dp, bool logscale)const{
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZeroInflatedLognormalModel::pdf(const Data * dp, bool logscale)const{
    double ans = logp(dynamic_cast<const DoubleData *>(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZeroInflatedLognormalModel::logp(double x)const{
    check_log_probabilities();
    if(x < precision_) return log_probability_of_zero_;
    return log_probability_of_positive_ + dlnorm(x, mu(), sigma(), true);
  }

  double ZeroInflatedLognormalModel::sim()const{
    if(runif() < positive_probability()){
      return exp(rnorm(mu(), sigma()));
    }
    return 0;
  }

  void ZeroInflatedLognormalModel::add_data(Ptr<Data> dp){
    if(dp->missing()) return;
    Ptr<DoubleData> d = DAT(dp);
    double y = d->value();
    add_data_raw(y);
  }

  void ZeroInflatedLognormalModel::add_data_raw(double y){
    if(y < precision_){
      binomial_->suf()->update_raw(0.0);
    }else{
      binomial_->suf()->update_raw(1.0);
      gaussian_->suf()->update_raw(log(y));
    }
  }

  void ZeroInflatedLognormalModel::add_mixture_data(Ptr<Data> dp, double prob){
    if(dp->missing()) return;
    double y = DAT(dp)->value();
    add_mixture_data_raw(y, prob);
  }

  void ZeroInflatedLognormalModel::add_mixture_data_raw(double y, double prob){
    if(y > precision_){
      gaussian_->suf()->add_mixture_data(log(y), prob);
      binomial_->suf()->add_mixture_data(1.0, prob);
    }else{
      binomial_->suf()->add_mixture_data(0, prob);
    }
  }

  void ZeroInflatedLognormalModel::clear_data(){
    gaussian_->clear_data();
    binomial_->clear_data();
  }

  void ZeroInflatedLognormalModel::combine_data(const Model &rhs, bool just_suf){
    const ZeroInflatedLognormalModel * rhsp =
        dynamic_cast<const ZeroInflatedLognormalModel *>(&rhs);
    if(!rhsp){
      ostringstream err;
      err << "ZeroInflatedLognormalModel::combine_data was called with an argument "
          << "that was not coercible to ZeroInflatedLognormalModel." << endl;
      report_error(err.str());
    }
    gaussian_->combine_data( *(rhsp->gaussian_), true);
    binomial_->combine_data( *(rhsp->binomial_), true);
  }

  void ZeroInflatedLognormalModel::find_posterior_mode(){
    gaussian_->find_posterior_mode();
    binomial_->find_posterior_mode();
  }

  void ZeroInflatedLognormalModel::mle(){
    gaussian_->mle();
    binomial_->mle();
  }

  void ZeroInflatedLognormalModel::set_conjugate_prior(
      double normal_mean_guess,
      double normal_mean_weight,
      double normal_standard_deviation_guess,
      double normal_standard_deviation_weight,
      double nonzero_proportion_guess,
      double nonzero_proportion_weight){
    gaussian_->set_conjugate_prior(normal_mean_guess,
                                   normal_mean_weight,
                                   normal_standard_deviation_weight,
                                   normal_standard_deviation_guess);
    double a = nonzero_proportion_weight * nonzero_proportion_guess;
    double b = nonzero_proportion_weight * (1 - nonzero_proportion_guess);
    binomial_->set_conjugate_prior(a, b);
    NEW(ZeroInflatedLognormalPosteriorSampler, sampler)(this);
    this->set_method(sampler);
  }

  double ZeroInflatedLognormalModel::mu()const{
    return gaussian_->mu();}
  void ZeroInflatedLognormalModel::set_mu(double mu){
    gaussian_->set_mu(mu);}

  double ZeroInflatedLognormalModel::sigma()const{
    return gaussian_->sigma();}
  void ZeroInflatedLognormalModel::set_sigma(double sigma){
    gaussian_->set_sigsq(sigma * sigma);}
  void ZeroInflatedLognormalModel::set_sigsq(double sigsq){
    gaussian_->set_sigsq(sigsq);}

  double ZeroInflatedLognormalModel::positive_probability()const{
    return binomial_->prob();}
  void ZeroInflatedLognormalModel::set_positive_probability(double prob){
    return binomial_->set_prob(prob);}

  double ZeroInflatedLognormalModel::mean()const{
    return positive_probability() * exp(mu() + .5 * gaussian_->sigsq());
  }

  double ZeroInflatedLognormalModel::variance()const{
    double sigsq = gaussian_->sigsq();
    return (exp(sigsq) - 1) * exp(2 * mu() + sigsq);
  }

  double ZeroInflatedLognormalModel::sd()const{
    return sqrt(variance());
  }

  Ptr<GaussianModel> ZeroInflatedLognormalModel::Gaussian_model(){
    return gaussian_;}

  Ptr<BinomialModel> ZeroInflatedLognormalModel::Binomial_model(){
    return binomial_;}


  Ptr<DoubleData> ZeroInflatedLognormalModel::DAT(Ptr<Data> dp)const{
    if(!!dp) return dp.dcast<DoubleData>();
    return Ptr<DoubleData>();
  }

  boost::function<void(void)>
      ZeroInflatedLognormalModel::create_binomial_observer(){
    return boost::bind(&ZeroInflatedLognormalModel::observe_binomial_probability, this);
  }

  void ZeroInflatedLognormalModel::observe_binomial_probability(){
    log_probabilities_are_current_ = false;
  }

  void ZeroInflatedLognormalModel::check_log_probabilities()const{
    if(log_probabilities_are_current_) return;
    log_probability_of_positive_ = log(positive_probability());
    log_probability_of_zero_ = log(1 - positive_probability());
    log_probabilities_are_current_ = true;
  }
}
