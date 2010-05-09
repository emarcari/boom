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
#include <Models/PosteriorSamplers/BetaBinomialSampler.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef BetaBinomialSampler BBS;

  BBS::BetaBinomialSampler(Ptr<BinomialModel> mod,
			   Ptr<BetaModel> pri)
    : mod_(mod),
      pri_(pri)
  {}

  void BBS::draw(){
    double a = pri_->a();
    double b = pri_->b();
    double nyes = mod_->suf()->sum();
    double n = mod_->n() * mod_->suf()->nobs();
    double nno =n - nyes;
//     double y1 = rgamma_mt(rng(), a+nyes, 1.0);
//     double y2 = rgamma_mt(rng(), b+nno, 1.0);
//     double p = y1/(y1+y2);
    double p = rbeta_mt(rng(), a + nyes, b+nno);
    mod_->set_prob(p);
  }

  double BBS::logpri()const{
    double p = mod_->prob();
    return pri_->logp(p);
  }

}
