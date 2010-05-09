#ifndef BOOM_SCALAR_HOMOGENEOUS_STATE_SPACE_MODEL_HPP
#define BOOM_SCALAR_HOMOGENEOUS_STATE_SPACE_MODEL_HPP
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

#include <Models/StateSpace/HomogeneousStateModel.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/TimeSeries/TimeSeriesDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/StateSpace/Filters/ScalarHomogeneousKalmanFilter.hpp>
#include <Models/MvnModel.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>


namespace BOOM{

  class ScalarHomogeneousStateSpaceModel
      : public CompositeParamPolicy,
        public TimeSeriesDataPolicy<DoubleData>,
        public PriorPolicy
  {

    // y[t] = Z^T alpha[t] + N(0,sigsq);
    // alpha[t] = T alpha[t-1] + R N(0,V)
    // dim(V) <= dim(alpha)

    // Z, T and R may depend on parameters, but do not change over time

  public:
    typedef HomogeneousStateModel StateT;

    ScalarHomogeneousStateSpaceModel(double sigsq=1.0);
    ScalarHomogeneousStateSpaceModel(const Vec &y);
    ScalarHomogeneousStateSpaceModel(std::vector<Ptr<StateT> >);
    ScalarHomogeneousStateSpaceModel(const ScalarHomogeneousStateSpaceModel & rhs);
    ScalarHomogeneousStateSpaceModel * clone()const;

    ScalarHomogeneousStateSpaceModel * add_state(Ptr<StateT>);

    Mat filter(const Vec & ts)const;
    Mat smooth(const Vec & ts)const;
    Mat state_mean_smooth(const Vec & ts)const;

    Mat filter(const TimeSeries<DoubleData> &)const;
    Mat smooth(const TimeSeries<DoubleData> &)const;


    Vec forecast(uint n)const;   // next n periods after data used to fit the model
    Vec forecast(uint n, const Vec & history)const;
    Vec forecast(uint n, const TimeSeries<DoubleData> & )const;

    uint total_state_size()const;
    uint total_innovation_size()const;

    double impute_state();  // returns log likelihood
    void sample_state_posterior();
    void set_obs_sigsq(double sigsq);
    double obs_sigsq()const;

    const Vec & Z()const;
//     const Mat & T()const;
//     const Mat & R()const;
//     const Spd & Q();

    double pdf(Ptr<Data>, bool logscale=false)const;
    double logp(const Vec & ts)const;
    void set_obs_filename(const string & );

    void observe_state(const Vec & now, const Vec & next);

    double obs_sum_of_squares()const;
    double obs_df()const;

    Vec simulate_initial_state()const;
    double log_state_model_prior()const;

    Ptr<MvnModel> initial_state_distribution()const;
    void save_state(bool do_it=true);

    const std::vector<Mat> & state()const;
    const std::vector<Mat> & state(const TimeSeries<DoubleData> &)const;

    void set_filter(); // gets everything set up to do filtering and smoothing
  private:
    Ptr<ZeroMeanGaussianModel> obs_;

    mutable Ptr<ScalarHomogeneousKalmanFilter> filter_;
    std::vector<Ptr<StateT> > state_;
    uint total_state_size_;
    uint total_innovation_size_;  // <= total state size

    Vec Z_;
    Mat T_;
    Mat R_;
    Spd Q_;

    GaussianSuf suf_;   // for keeping track of prediction errors

    mutable Ptr<MvnModel> init_;
    Vec make_Z();
    Mat make_T();
    Mat make_R();
    Spd make_Q();

    void clear_state_models();

    bool save_state_;
    typedef std::map<const TimeSeries<DoubleData> *, std::vector<Mat> > StateStorage;
    StateStorage state_storage_;
  };



}

#endif// BOOM_SCALAR_HOMOGENEOUS_STATE_SPACE_MODEL_HPP
