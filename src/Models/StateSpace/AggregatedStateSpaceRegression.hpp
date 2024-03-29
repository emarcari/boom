/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_AGGREGATED_STATE_SPACE_REGRESSION_HPP_
#define BOOM_AGGREGATED_STATE_SPACE_REGRESSION_HPP_

#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/StateSpaceRegressionModel.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <Models/StateSpace/Filters/SparseKalmanTools.hpp>
#include <Models/StateSpace/Filters/ScalarKalmanStorage.hpp>
#include <Models/StateSpace/StateModels/RegressionStateModel.hpp>

#include <Models/GaussianModel.hpp>
#include <Models/Glm/RegressionModel.hpp>

#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

#include <boost/scoped_ptr.hpp>

// AggregatedStateSpaceRegression models a StateSpaceModel that
// produces values on a fine time scale (e.g. weeks) but is only
// observed on a coarse time scale (e.g. months).  The model can be
// written down as
//
//       w[t] ~ N(Z[t].dot(alpha[t]), client_observation_variance)
// alpha[t+1] ~ N(T[t] * alpha[t], client_state_variance)
//
// Moving w[t] into the state gives
// alpha[t+1] ~ N(T[t] * alpha[t], client_state_variance)
//     w[t+1] =  Z[t+1]^T T[t+1] * alpha[t] + Z[t+1]^T R[t] * eta[t] + eps[t+1]
//     W[t+1] = (1-delta[t]) * W[t] + (1-delta[t]*phi[t]) * w[t]

// The state in this model is (a[t], w[t], W[t]), where a[t] is the
// state from the underlying StateSpaceRegressionModel, w[t] is the
// amount produced by the underlying model for period [t], and W[t] is
// the cumulative amount produced in the current month, before period
// [t].  Let phi[t] be the fraction of the current week's output
// belonging to the month containing the first day in the week.  Then
// Y[t] = W[t] + phi[t]*w[t] is the current total, for the month
// containing the start of week t.  Let delta[t] denote the indicator
// that a week contains the end of an existing month.

// The transition matrix is:
// |T[t]            0                 0     |
// |Z[t+1]'T[t]     0                 0     |
// |0        1-phi[t]*delta[t] (1-delta[t]) |
//
// Note that Z is from time t+1, while the others are from time t.
//
// The observation matrix is (0 0 0 0 ... phi 1).  This is not the
// same as 'Z', which is the observation matrix for the underlying
// state space model.

namespace BOOM{
  //======================================================================
  // FineNowcastingData describes one 'week' in the life of a
  // nowcasting model.  We observe the vector of signals for that week
  // and if the week contains the start of a new month then we observe
  // the monthly total for the month that just passed.  If a week
  // straddles monthly boundaries, we need to know how much of that
  // week's activity is attributable to the earlier of the two months
  // that contains it.
  //
  // Note that 'months' and 'weeks' are conceptual placeholders for
  // coarse- and fine-grained time intervals.  'months' might be weeks
  // and 'weeks' might be days, for example, or 'months' might be
  // quarters, and 'weeks' might be months.
  class FineNowcastingData : public Data {
   public:
    // Args:
    //   x:  The vector of observables from this week.
    //   coarse_observation: The observed value for the month
    //     containing the _start_ of this week.  This will frequently
    //     be unobserved, in which case an arbitrary value can be
    //     assigned.
    //   coarse_observation_observed: True if this week contains the
    //     end of a month, and the monthly total is known.
    //   contains_end:  True if this week contains the end of a month.
    //   fraction_in_initial_period: The fraction of this week's
    //     output belonging to the month containing the _start_ of the
    //     week.  This is always positive, and usually 1.
    //
    // Note that 'contains_end' and 'coarse_observation_observed' are
    // almost redundant.  However, 'coarse_observation_observed' can
    // be false when a new month begins if the release of the most
    // recent monthly totals is delayed (e.g. if the coarse data is
    // not as up-to-date as the fine data).
    FineNowcastingData(const Vec &x,
                         double coarse_observation,
                         bool coarse_observation_observed,
                         bool contains_end,
                         double fraction_in_initial_period);
    FineNowcastingData(const FineNowcastingData &rhs);
    virtual FineNowcastingData * clone()const;
    virtual ostream &display(ostream &out)const;
    virtual uint size(bool minimal)const;

    Ptr<RegressionData> regression_data()const;
    double fraction_in_initial_period()const;
    bool contains_end()const;
    bool coarse_observation_observed()const;
    double coarse_observation()const;

   private:
    Ptr<RegressionData> x_;
    double coarse_observation_;
    bool coarse_observation_observed_;
    bool contains_end_;
    double fraction_in_initial_period_;
  };

  //======================================================================
  class AccumulatorTransitionMatrix : public SparseKalmanMatrix{
   public:
    // If this matrix is for the transition from time t to time t+1 then
    // Args:
    //   T_t: the client state transition matrix at time t. (For the
    //     t->t+1 transistion)
    //   Z_t_plus_1: The client model observation vector at time t+1.
    //   contains_end: Indicates whether week t contains the end of
    //     a month.
    //   fraction_in_initial_period: Proportion of output in week t
    //     attributed to the month containing the start of week t.
    //   owns_matrix: If true then this class will take ownership of
    //     T, which will be deleted by the destructor.
    AccumulatorTransitionMatrix(
        const SparseKalmanMatrix *T_t,
        const SparseVector &Z_t_plus_1,
        double fraction_in_initial_period,
        bool contains_end,
        bool owns_matrix = false);

    ~AccumulatorTransitionMatrix();

    // Resets the class as if the constructor had been called with
    // these arguments.
    void reset(const SparseKalmanMatrix *T,
               const SparseVector &Z,
               double fraction_in_initial_period,
               bool contains_end);

    // Number of rows and columns in the augmented matrix (i.e. after
    // adding w and W to the state).
    virtual int nrow()const;
    virtual int ncol()const;

    virtual Vec operator*(const Vec &v)const;
    virtual Vec operator*(const VectorView &v)const;
    virtual Vec operator*(const ConstVectorView &v)const;

    virtual Vec Tmult(const Vec &v)const;
    virtual void sandwich_inplace(Spd &P)const;
    virtual Mat & add_to(Mat &P)const;
   private:
    const SparseKalmanMatrix * transition_matrix_;
    SparseVector observation_vector_;
    double fraction_in_initial_period_;
    bool contains_end_;
    bool owns_matrix_;
  };

  //======================================================================
  // The accumulator state is (alpha, w, W)^T.  The errors look like
  //
  //             R[t] eta[t]                   (normal state error)
  //    Z[t+1]^T R[t] eta[t] + epsilon[t+1]    (fine error)
  //                   0                       (error in cumulator)
  //
  // This means that the "R matrix" and the "error vector" are
  // | R    0 |  | eta     |
  // | Z^TR 1 |  | epsilon |
  // | 0    0 |
  //
  // This class describes the rank deficient variance matrix of the
  // initial matrix product.
  class AccumulatorStateVarianceMatrix : public SparseKalmanMatrix{
   public:
    // When calling the constructor, the observation matrix and
    // observation variance should be with respect to time t+1, while
    // RQR is with respect to time t.
    AccumulatorStateVarianceMatrix(
        const SparseKalmanMatrix *RQR,
        const SparseVector &Z,
        double observation_variance,
        bool owns_matrix = false);
    ~AccumulatorStateVarianceMatrix();

    void reset(const SparseKalmanMatrix *RQR,
               const SparseVector &Z,
               double observation_variance);

    virtual int nrow()const;
    virtual int ncol()const;

    virtual Vec operator*(const Vec &v)const;
    virtual Vec operator*(const VectorView &v)const;
    virtual Vec operator*(const ConstVectorView &v)const;

    virtual Vec Tmult(const Vec &x)const;
    virtual Mat & add_to(Mat &P)const;
   private:
    // See the fragility comments in AccumulatorTransitionMatrix.
    const SparseKalmanMatrix *state_variance_matrix_;
    SparseVector observation_vector_;
    double observation_variance_;
    bool owns_matrix_;
  };

  //======================================================================
  // This class is-a RegressionStateModel with an extra dummy
  // predictor that can be returned at one past the final data point.
  //
  // The override of RegressionStateModel is necessary because
  // the transition matrix for time t depends on Z[t+1], which
  // includes a predictor vector for time t+1.  The shifted predictor
  // is a problem for the last time point, for which no predictor has
  // been observed.  The final time point makes no difference to the
  // Kalman simulation smoother, but it does forecast alpha[t+1].
  class AggregatedRegressionStateModel
      : public RegressionStateModel {
   public:
    AggregatedRegressionStateModel(Ptr<RegressionModel> m);
    void set_final_x(const Vec &x);
    virtual SparseVector observation_matrix(int t)const;
   private:
    Vec final_x_;
  };

  //======================================================================
  // AggregatedStateSpaceRegression models a time series of
  // FineNowcastingData.
  class AggregatedStateSpaceRegression
      : public StateSpaceModelBase,
        public IID_DataPolicy<FineNowcastingData>,
        public PriorPolicy
  {
   public:
    AggregatedStateSpaceRegression(int number_of_predictors);
    AggregatedStateSpaceRegression(const AggregatedStateSpaceRegression &rhs);
    virtual AggregatedStateSpaceRegression * clone()const;

    // Need to override add_data so that x's can be shared with the
    // regression model.
    virtual void add_data(Ptr<Data>);
    virtual void add_data(Ptr<FineNowcastingData>);

    // A shortcut data accessor emphasizing that data point t is
    // measured on the 'weekly' scale, rather than 'monthly'.
    Ptr<FineNowcastingData> fine_data(int t);
    const Ptr<FineNowcastingData> fine_data(int t)const;

    // Number of fine scale time points ('weeks') in the training data.
    int time_dimension()const;

    // Number of elements in the state vector at a single time point,
    // including the cumulator variables.  Because the cumulator
    // variables are not accounted for by a StateModel, we need to
    // overload this function and account for them by hand.
    virtual int state_dimension()const;

    // Variance of observed data w[t], given state alpha[t].  Durbin
    // and Koopman's H.  For this model, this is 0.
    virtual double observation_variance(int t)const;

    // An adjusted_observation(t) is w[t] after subtracting off
    // regression effects not accounted for in the state model.  We
    // can just return the W[t] portion of the week[t], if observed
    // because our regression adjustment is already part of the state
    // vector.
    virtual double adjusted_observation(int t)const;

    // Returns an indicator of whether W[t] is observed for week t.
    virtual bool is_missing_observation(int t)const;

    // The regression model is the observation model for the
    // non-cumulated client data.  The observation model for the
    // cumulated data is a Gaussian model with zero variance.
    virtual GaussianModel * observation_model(){
      return observation_model_.get();}
    virtual const GaussianModel * observation_model()const{
      return observation_model_.get(); }

    // Returns a pointer to the RegressionModel that manages the
    // linear prediction based on contemporaneous covariates.
    RegressionModel * regression_model(){
      return regression_.get();}
    const RegressionModel * regression_model()const{
      return regression_.get();}

    // Returns a pointer to the RegressionStateModel used in the
    // Kalman filter.
    AggregatedRegressionStateModel * regression_state_model() {
      return regression_state_.get();}
    const AggregatedRegressionStateModel * regression_state_model()const{
      return regression_state_.get();}

    // This function updates the regression portion of the model.
    virtual void observe_data_given_state(int t);

    virtual const AccumulatorTransitionMatrix *
    state_transition_matrix(int t)const;

    virtual SparseVector observation_matrix(int t)const;

    virtual const AccumulatorStateVarianceMatrix *
    state_variance_matrix(int t)const;

    virtual void simulate_initial_state(VectorView v)const;
    virtual Vec simulate_initial_state()const;
    virtual Vec simulate_state_error(int t)const;

    virtual Vec initial_state_mean()const;
    virtual Spd initial_state_variance()const;

   private:
    Ptr<RegressionModel> regression_;
    std::vector<Ptr<StateModel> > state_models_;
    Ptr<GaussianModel> observation_model_;
    Ptr<AggregatedRegressionStateModel> regression_state_;
    mutable boost::scoped_ptr<AccumulatorStateVarianceMatrix> variance_matrix_;
    mutable boost::scoped_ptr<AccumulatorTransitionMatrix> transition_matrix_;
  };

}
#endif // BOOM_AGGREGATED_STATE_SPACE_REGRESSION_HPP_
