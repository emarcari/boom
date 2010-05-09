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
#include "GaussianMeanSampler.hpp"
#include <Models/GaussianModel.hpp>
#include <cmath>
#include <distributions.hpp>

namespace BOOM{
  typedef GaussianMeanSampler GMS;

  GMS::GaussianMeanSampler(Ptr<GaussianModel> Mod, double mu, double tau)
    : mu_(Mod->Mu_prm()),
      mod(Mod),
      pri(new GaussianModel(mu, tau))
  {}

  GMS::GaussianMeanSampler(Ptr<GaussianModel> Mod, Ptr<GaussianModel> Pri)
    : mu_(Mod->Mu_prm()),
      mod(Mod),
      pri(Pri)
  {}

  double GMS::logpri()const{
    return pri->pdf(mu_, true);  }

  void GMS::draw(){
    Ptr<GaussianSuf> s = mod->suf();

    double ybar = s->ybar();
    double n = s->n();

    double sigsq = mod->sigsq();

    double mu0 = pri->mu();
    double tausq = pri->sigsq();

    double ivar= (n/sigsq) + (1.0/tausq);
    double mean = (n*ybar/sigsq + mu0/tausq)/ivar;
    double sd = sqrt(1.0/ivar);

    double ans = rnorm_mt(rng(), mean, sd);
    mu_->set(ans);
  }
}
