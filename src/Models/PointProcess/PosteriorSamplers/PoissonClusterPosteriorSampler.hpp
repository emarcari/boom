/*
  Copyright (C) 2005-2012 Steven L. Scott

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
#ifndef BOOM_POISSON_CLUSTER_POSTERIOR_SAMPLER_HPP_
#define BOOM_POISSON_CLUSTER_POSTERIOR_SAMPLER_HPP_

namespace BOOM{
  class PoissonClusterPosteriorSampler : public PosteriorSampler{
   public:
    PoissonClusterPosteriorSampler(PoissonClusterProcess *model)
        : model_(model)
    {}
    virtual void draw(){
      model_->impute_latent_data(rng());
      model_->sample_client_posterior();
    }
    virtual double logpri()const{
      return model_->logpri();
    }
   private:
    PoissonClusterProcess *model_;
  };
}

#endif // BOOM_POISSON_CLUSTER_POSTERIOR_SAMPLER_HPP_
