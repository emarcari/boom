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
#include <Models/PosteriorSamplers/MultinomialDirichletSampler.hpp>
#include <Models/MultinomialModel.hpp>
#include <Models/DirichletModel.hpp>
#include <distributions.hpp>

namespace BOOM{
  typedef MultinomialDirichletSampler MDS;
  typedef MultinomialModel MM;
  typedef DirichletModel DM;

  MDS::MultinomialDirichletSampler(Ptr<MM> Mod, const Vec &Nu)
    : mod(Mod),
      pri(new DM(Nu))
  {}

  MDS::MultinomialDirichletSampler(Ptr<MM> Mod, Ptr<DM> Pri)
    : mod(Mod),
      pri(Pri)
  {}

  MDS::MultinomialDirichletSampler(const MDS &rhs)
    : PosteriorSampler(rhs),
      mod(rhs.mod->clone()),
      pri(rhs.pri->clone())
  {}

  MDS * MDS::clone()const{return new MDS(*this);}

  void MDS::draw(){
    Vec counts = pri->nu() +  mod->suf()->n();
    Vec pi = rdirichlet_mt(rng(), counts);
    mod->set_pi(pi);
  }
  void MDS::find_posterior_mode(){
    Vec counts = pri->nu() +  mod->suf()->n();
    Vec pi = mdirichlet(counts);
    mod->set_pi(pi);
  }

  double MDS::logpri()const{
    return pri->logp(mod->pi());
  }

}
