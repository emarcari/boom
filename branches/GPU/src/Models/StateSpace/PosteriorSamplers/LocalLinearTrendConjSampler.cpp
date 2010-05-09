/*
  Copyright (C) 2008 Steven L. Scott

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

#include <Models/StateSpace/PosteriorSamplers/LocalLinearTrendConjSampler.hpp>
#include <Models/GammaModel.hpp>
#include <Models/StateSpace/LocalLinearTrend.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef LocalLinearTrendConjSampler LLTCS;

  LLTCS::LocalLinearTrendConjSampler(LocalLinearTrend *m, Ptr<GammaModelBase> pri)
      : mod_(m),
        pri_(pri)
  {}

  void LLTCS::draw(){
    double a = pri_->alpha() + mod_->suf().n() * .5;
    double b = pri_->beta() + mod_->suf().sumsq() * .5;
    double siginv = rgamma(a,b);
    mod_->set_sigsq(1.0/siginv);
  }

  double LLTCS::logpri()const{
    double siginv = 1.0/mod_->sigsq();
    return pri_->logp(siginv);
  }

  void LLTCS::find_posterior_mode(){
    double a = pri_->alpha() + mod_->suf().n() * .5;
    double b = pri_->beta() + mod_->suf().sumsq() * .5;
    mod_->set_sigsq(b/(a-1.0));
  }

}
