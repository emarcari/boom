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
#ifndef BOOM_SEP_STRAT_SIGMA_SAMPLER_HPP
#define BOOM_SEP_STRAT_SIGMA_SAMPLER_HPP

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/ParamTypes.hpp>
namespace BOOM{

  class SepStratSigmaSampler
    : public PosteriorSampler
  {
  public:
    SepStratSigmaSampler(Ptr<SpdParams> Sigma);
    virtual double logpri()const;
    virtual void draw();
    void set_sumsq(const Spd &S);
  private:
    double loglike()const;
    void draw_pos(uint i, uint j);
    void set_params();
    double log_det_R()const;
    double log_det_Rinv()const;

    Spd Sumsq;
    Ptr<SpdParams> sigma_;

    Spd R;
    Spd Rinv;

    Vec S;
    Vec Sinv;

    Mat L;
    Mat Linv;
  };

}
#endif// BOOM_SEP_STRAT_SIGMA_SAMPLER_HPP
