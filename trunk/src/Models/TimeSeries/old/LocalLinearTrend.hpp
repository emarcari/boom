/*
  Copyright (C) 2005 Steven L. Scott

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
#include <BOOM.hpp>
#include <LinAlg/Types.hpp>
#include <Models/TimeSeries/StateSpaceModel.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/TimeSeries/TimeSeriesDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{

  class GaussianModel;

  class LocalLinearTrend
    : public StateSpaceModel,
      public CompositeParamPolicy,
    //      public TimeSeriesDataPolicy<StateSpaceData>,
      public PriorPolicy
  {
    LocalLinearTrend();  // variances are 1
    LocalLinearTrend(double sig_obs, double sig_state);
    LocalLinearTrend(const std::vector<double> &);
    LocalLinearTrend(const LocalLinearTrend &);
    LocalLinearTrend * clone()const;

    Ptr<UnivParams> Sigsq_obs();
    const Ptr<UnivParams> Sigsq_obs()const;
    Ptr<UnivParams> Sigsq_state();
    const Ptr<UnivParams> Sigsq_state()const;

    const double & sigsq_obs()const;
    void set_sigsq_obs(double);

    const double & sigsq_state()const;
    void set_sigsq_state(double);

    virtual void add_data(Ptr<Data>);
    virtual void add_data(Ptr<StateSpaceData>);
    virtual void add_data(Ptr<ScalarStateSpaceData>);
    virtual void add_data(Ptr<DataSeriesType>);
    void AddData(double);

    Ptr<ScalarLatentState> STATE(Ptr<LatentState>)const;
  private:
    void setup_params();
    Ptr<GaussianModel> obs_err_;
    Ptr<GaussianModel> state_err_;
  };
}
