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

#ifndef BOOM_BETA_BINOMIAL_SAMPLER_HPP
#define BOOM_BETA_BINOMIAL_SAMPLER_HPP

#include <Models/BetaModel.hpp>
#include <Models/BinomialModel.hpp>

namespace BOOM{

  class BetaBinomialSampler
    : public PosteriorSampler
  {
  public:
    BetaBinomialSampler(BinomialModel *, Ptr<BetaModel>);
    virtual void draw();
    virtual double logpri()const;
    void find_posterior_mode();
  private:
    BinomialModel *mod_;
    Ptr<BetaModel> pri_;
  };
}

#endif// BOOM_BETA_BINOMIAL_SAMPLER_HPP
