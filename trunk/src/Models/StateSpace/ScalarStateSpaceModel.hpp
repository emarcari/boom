#ifndef BOOM_SCALAR_STATE_SPACE_MODEL_HPP
#define BOOM_SCALAR_STATE_SPACE_MODEL_HPP
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

#include <Models/StateSpace/StateModel.hpp>


namespace BOOM{

  class ScalarStateSpaceModel
    : public ScalarStateSpaceModel,
      public TimeSeriesDataPolicy(DoubleData)
  {

    // y[t] = Z^T alpha[t] + N(0,sigsq);
    // alpha[t] = T alpha[t-1] + R N(0,V)
    // dim(V) <= dim(alpha)

    // T and R do not change over time

  public:
    typedef StateModel StateT;
    typedef ScalarObservationModel StateT;

    ScalarSateSpaceModel();
    ScalarSateSpaceModel(Ptr<ObsT>);
    ScalarSateSpaceModel(Ptr<ObsT>, std::vector<Ptr<StateT> >);

    ScalarSateSpaceModel(const ScalarSateSpaceModel & rhs);
    ScalarSateSpaceModel * clone()const;

    ScalarSateSpaceModel * add_state_model(Ptr<StateT>);

    Vec forecast(uint n)const;   // next n periods after data used to fit the model
    Vec forecast(uint n, const Vec & history)const;
    Vec forecast(uint n, const TimeSeries<DoubleData>)const;

    void impute_state();
  
  private:
    Ptr<ScalarKalmanFilter> filter_;
    Ptr<ObsT> obs_;
    std::vector<Ptr<StateT> > state_;
    std::vector<uint> > state_size_;
    uint total_state_size_;
    Mat T_;
    Mat R;

    Mat make_T();
    Mat make_R();
    Vec make_Z();

  };



}

#endif// BOOM_SCALAR_STATE_SPACE_MODEL_HPP
