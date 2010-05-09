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

#ifndef BOOM_STATE_SPACE_LOCAL_LINEAR_TREND_HPP
#define BOOM_STATE_SPACE_LOCAL_LINEAR_TREND_HPP

#include <Models/StateSpace/HomogeneousStateModel.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/GaussianModelBase.hpp>
#include <Models/StateSpace/PosteriorSamplers/LocalLinearTrendConjSampler.hpp>

namespace BOOM{

  class LocalLinearTrend
      : public HomogeneousStateModel,
        public ParamPolicy_1<UnivParams>,
        public ConjugatePriorPolicy<LocalLinearTrendConjSampler>
  {

    // state is (nu, mu)
    // nu is the slope, and mu is the current level
    //
    // model is nu[t+1] = nu[t] + error
    //          mu[t+1] = mu[t] + nu[t+1] = mu[t] + nu[t] + error
    // Z = (0,1)
    // Q = sigsq  (1x1)
    // R = (1)
    //     (1)
    // T = (1 0)
    //     (1 1)

   public:
    LocalLinearTrend(double sigsq = 1);

    virtual LocalLinearTrend * clone()const;
    virtual void observe_state(const ConstVectorView & now,
                               const ConstVectorView & next);
    virtual uint state_size()const;
    virtual uint innovation_size()const;
    virtual void clear_data();
    virtual void combine_data(const Model &, bool just_suf=true);

    const GaussianSuf & suf();
    double sigsq()const;
    void set_sigsq(double sigsq);  // must update Q
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm()const;
   private:
    GaussianSuf suf_;
    virtual void add_data(Ptr<Data>);
    virtual double pdf(Ptr<Data> , bool logscale=false)const;
  };
}
#endif // BOOM_STATE_SPACE_LOCAL_LINEAR_TREND_HPP
