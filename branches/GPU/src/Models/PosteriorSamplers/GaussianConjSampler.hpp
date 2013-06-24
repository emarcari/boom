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

#ifndef BOOM_GAUSSIAN_MODEL_CONJUGATE_SAMPLER_HPP
#define BOOM_GAUSSIAN_MODEL_CONJUGATE_SAMPLER_HPP

#include <Models/GaussianModel.hpp>
#include <Models/GaussianModelGivenSigma.hpp>
#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM{
  class GaussianConjSampler
    : public PosteriorSampler
  {
  public:
    GaussianConjSampler(Ptr<GaussianModel> m,
			Ptr<GaussianModelGivenSigma> mu,
			Ptr<GammaModel> sig);
    void draw();
    double logpri()const;

    double mu()const;
    double kappa()const;
    double df()const;
    double ss()const;

    void find_posterior_mode();
  private:
    Ptr<GaussianModel> mod_;
    Ptr<GaussianModelGivenSigma> mu_;
    Ptr<GammaModel> siginv_;
  };
}
#endif// BOOM_GAUSSIAN_MODEL_CONJUGATE_SAMPLER_HPP