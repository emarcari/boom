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
#include <Models/StateSpace/PosteriorSamplers/ScalarSsmPosteriorSampler.hpp>
#include <distributions.hpp>
#include <Models/StateSpace/ScalarHomogeneousStateSpaceModel.hpp>

namespace BOOM{

  typedef ScalarSsmPosteriorSampler SSPS;

  SSPS::ScalarSsmPosteriorSampler(ScalarHomogeneousStateSpaceModel * mod,
                                  Ptr<GammaModelBase> ivar,
                                  Ptr<MvnGivenSigma> a0,
                                  Ptr<WishartModel> P0)
    : m_(mod),
      ivar_(ivar)
  {
    Ptr<MvnModel> init(mod->initial_state_distribution());
    init_prior_ = new MvnConjSampler(init, a0, P0);
  }

  void SSPS::draw(){
    m_->impute_state();
    double ss = m_->obs_sum_of_squares();
    double df = m_->obs_df();

    double a = ivar_->alpha() + df/2.0;
    double b = ivar_->beta() + ss/2.0;

    double siginv = rgamma(a,b);
    m_->set_obs_sigsq(1.0/siginv);

    m_->sample_state_posterior();

    init_prior_->draw();
  }

  double SSPS::logpri()const{
    double siginv  = 1.0/m_->obs_sigsq();
    double ans = ivar_->logp(siginv);
    ans += m_->log_state_model_prior();
    return ans;
  }
}
