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

#ifndef BOOM_TREG_TSE_HPP    // t-regression with time series errors
#define BOOM_TREG_TSE_HPP

#include <Models/glm/TRegression.hpp>
#include <Models/TimeSeries/WeightedSeasonalLocalLevelModel.hpp>

namespace BOOM{

  class TregTseData:
    public WeightedStateSpaceData,
    public WeightedRegressionData
  {
    double y_actual;
    Ptr<WeightedStateSpaceData> ssd_;
    Ptr<WeightedRegressionData> rd_;
    friend TregTseModel;
  public:
    TregTseData();
    TregTseData(double Y, const Vec &X, uint StateSize);
    TregTseData(double Y, const Vec &X, Ptr<TregTseData> Prev);
    TregTseData(const TregTseData &rhs);
    TregTseData * clone()const;

    ostream & display(ostream &)const;
    istream & read(istream &);
    uint size()const;

    void set_regression(Ptr<StateSpaceModel> m);
    void set_ts(Ptr<RegressionModelBase> m);
  };

  //----------------------------------------------------------------------

  class TregTseModel
    : public CompositeParamPolicy,
      public TimeSeriesDataPolicy<TregTseData>,
      public PriorPolicy,
    public

  {
    Ptr<TRegressionModel> treg;
    Ptr<WeightedSeasonalLocalLevelModel> ts;

    void setup_data();  // make client models point to same data as full model
    void setsuf()const;
    void setprm()const;
  public:
    typedef DataPolicy::DataSeriesType DataSeriesType;
    typedef DataPolicy::data_point_type data_point_type;
    typedef WeightedSeasonalLocalLevelModel WSLLM;

    TregTseModel(Ptr<t_regression_model> , Ptr<WSLLM> );
    TregTseModel(const TregTseModel &rhs);
    TregTseModel * clone()const;

    void set_data(const DataSeriesType &d);

    void initialize_params();
    double pdf(Ptr<data>, bool logscale)const;
    void adopt_treg_variance();
    void adopt_ts_variance();

    void setup_regression(DataSeriesType &d);  //
    void setup_ts(DataSeriesType &d);
    void setup_regression();
    void setup_ts();

    void set_initial_state(const Vec &alpha);
    void set_initial_state_dist(const Vec &a, const Spd &V);
    void set_initial_state_dist(double mu, double sigsq);
    void forecast(DataSeriesType &, Ptr<StateSpaceData> d0)const;

    void printy(const string &fname)const;
    void write_state(ostream &out)const;

    void impute_states();
    void impute_treg_data();
  };

}// namespace BOOM

#endif// BOOM_TREG_TSE_HPP
