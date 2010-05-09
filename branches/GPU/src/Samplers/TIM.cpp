/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include <Samplers/TIM.hpp>
#include <Samplers/MH_Proposals.hpp>
#include <stdexcept>
#include <boost/bind.hpp>
#include <boost/function.hpp>

namespace BOOM{

  TIM::TIM(const BOOM::Target & logf,
           const BOOM::dTarget & dlogf,
           const BOOM::d2Target & d2logf,
           int dim,
           double nu)
      : MetropolisHastings(logf, 0),
        prop_(create_proposal(dim, nu)),
        f_(logf),
        df_(dlogf),
        d2f_(d2logf),
        mode_is_fixed_(0),
        mode_has_been_found_(0)
  {
    MetropolisHastings::set_proposal(prop_);
  }

  inline double TIM_empty_target(const Vec &){ return 1.0; }

  typedef boost::function<double(const Vec &,Vec &, Mat &,int)> FullTarget;

  TIM::TIM(FullTarget logf,
           int dim, double nu)
      : MetropolisHastings(TIM_empty_target, 0),
        prop_(create_proposal(dim,nu)),
        mode_is_fixed_(0),
        mode_has_been_found_(0)
  {
    f_ = boost::bind(logf, _1, g_, H_, 0);
    df_ = boost::bind(logf, _1, _2, H_, 1);
    d2f_ = boost::bind(logf, _1, _2, _3, 2);
    MetropolisHastings::set_target(f_);
    MetropolisHastings::set_proposal(prop_);
  }

  Vec TIM::draw(const Vec & old){
    if(!mode_has_been_found_ || !mode_is_fixed_){
      bool ok = locate_mode(old);
      if(!ok) report_failure(old);
    }
    return MetropolisHastings::draw(old);
  }

  void TIM::report_failure(const Vec &old){
    ostringstream err;
    double value = d2f_(old, g_, H_);
    err << "failed attempt to find mode in BOOM::TIM" << endl
        << "current parameter value is " << endl << old << endl
        << "target function value at this parameter is " << value << endl
        << "current gradient is " << g_ << endl
        << "hessian matrix is " << H_ << endl
        ;
    throw std::runtime_error(err.str());
  }

  void TIM::fix_mode(bool yn){ mode_is_fixed_ = yn;}

  bool TIM::locate_mode(const Vec & old){
    cand_ = old;
    g_ = old;
    H_.resize(old.size());
    try{
      max_nd2(cand_, g_, H_, f_, df_, d2f_);
    }catch(std::exception &e){
      return false;
    }catch(...){
      return false;
    }
    H_*= -1;
    mode_has_been_found_ = true;
    prop_->set_mu(cand_);
    prop_->set_ivar(H_);
    return true;
  }
  const Vec & TIM::mode()const{return prop_->mode();}
  const Spd & TIM::ivar()const{return prop_->ivar();}

  Ptr<MvtIndepProposal> TIM::create_proposal(int dim, double nu){
    Vec mu(dim);
    Spd Sigma(dim);
    Sigma.set_diag(1.0);
    return new MvtIndepProposal(mu, Sigma, nu);
  }
}
