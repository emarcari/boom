/*
  Copyright (C) 2006 Steven L. Scott

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
#include "MvnVarSampler.hpp"
#include <Models/MvnModel.hpp>
#include <distributions.hpp>
namespace BOOM{

  typedef MvnConjVarSampler MCVS;
  MCVS::MvnConjVarSampler(Ptr<MvnModel> m)
    : mvn(m),
      pdf(new UnivParams(0.0))
  {
    Spd sumsq = m->Sigma().Id();
    sumsq.set_diag(0.0);
    pss = new SpdParams(sumsq);
  }

  MCVS::MvnConjVarSampler(Ptr<MvnModel> m, double df, const Spd &sumsq)
    : mvn(m),
      pdf(new UnivParams(df)),
      pss(new SpdParams(sumsq))
  {}

  double MCVS::logpri()const{
    const Spd & siginv(mvn->siginv());
    return dWish(siginv, pss->var(), pdf->value(), true);
  }

  void MCVS::draw(){
    Ptr<MvnSuf> s = mvn->suf();
    double df = pdf->value() + s->n();
    Spd S = s->center_sumsq();
    S += pss->value();
    S = rWish(df, S.inv());
    Ptr<SpdParams> sp = mvn->Sigma_prm();
    sp->set_ivar(S);
  }

}
