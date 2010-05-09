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
#ifndef BOOM_MVN_VAR_SAMPLER_HPP
#define BOOM_MVN_VAR_SAMPLER_HPP
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/SpdParams.hpp>

namespace BOOM{
  class MvnModel;
  class MvnConjVarSampler : public PosteriorSampler{
    // assumes y~N(mu, Sigma), with mu|Sigma \norm(mu0, Sigma/kappa)
    // and Sigma^-1~W(df, SS)
  public:
    MvnConjVarSampler(Ptr<MvnModel>, double df, const Spd & SS);
    MvnConjVarSampler(Ptr<MvnModel>);
    double logpri()const;
    void draw();
  private:
    Ptr<MvnModel> mvn;
    Ptr<UnivParams> pdf;
    Ptr<SpdParams> pss;
  };

  class MvnVarSampler : public PosteriorSampler{
    // assumes y~N(mu, Sigma) with mu~N(mu0, Omega) and Sigma^-1~W(df, SS)
  public:
    double logpri()const;
    void draw();
  private:
    Ptr<MvnModel> mvn;
  };
}
#endif// BOOM_MVN_VAR_SAMPLER_HPP
