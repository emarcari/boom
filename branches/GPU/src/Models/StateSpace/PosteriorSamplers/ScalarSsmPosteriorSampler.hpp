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

#ifndef BOOM_STATE_SPACE_SCALAR_SSM_POSTERIOR_SAMPLER_HPP
#define BOOM_STATE_SPACE_SCALAR_SSM_POSTERIOR_SAMPLER_HPP

#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/MvnGivenSigma.hpp>
#include <Models/WishartModel.hpp>
#include <Models/PosteriorSamplers/MvnConjSampler.hpp>

namespace BOOM{

  class ScalarHomogeneousStateSpaceModel;
  class ScalarSsmPosteriorSampler
      : public PosteriorSampler
  {
    // this class is the driver for sampling
    //
   public:
    ScalarSsmPosteriorSampler(ScalarHomogeneousStateSpaceModel * m,
                              Ptr<GammaModelBase> pri_,
                              Ptr<MvnGivenSigma>,
                              Ptr<WishartModel>);
    virtual void draw();
    virtual double logpri()const;
   private:
    ScalarHomogeneousStateSpaceModel *m_;
    Ptr<GammaModelBase> ivar_;
    Ptr<MvnConjSampler> init_prior_;  // prior for initial state
  };

}
#endif // BOOM_STATE_SPACE_SCALAR_SSM_POSTERIOR_SAMPLER_HPP
