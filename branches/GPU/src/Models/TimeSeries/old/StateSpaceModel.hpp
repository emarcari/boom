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

#ifndef STATE_SPACE_MODEL_HPP
#define STATE_SPACE_MODEL_HPP

#include "StateSpaceData.hpp"
#include "TimeSeries.hpp"
#include "TimeSeriesDataPolicy.hpp"

namespace BOOM{


  class StateSpaceModel
    : public TimeSeriesDataPolicy<StateSpaceData, TimeSeries<StateSpaceData> >
  {
  public:
    typedef StateSpaceData ssd;
    typedef DataSeriesType DVEC;

    StateSpaceModel(const Mat &T, const Mat &Z, const Mat &R);
    StateSpaceModel(const StateSpaceModel &rhs);
    StateSpaceModel * clone()const=0;

    virtual double kalman_filter(DataSeriesType &)const;   // returns log likelihood
    virtual void disturbance_smoother(DataSeriesType &, bool eta=true, bool eps=false)const;
    virtual void state_mean_smoother(DataSeriesType &)const;
    virtual void kalman_smoother(DataSeriesType &)const;

    double loglike()const;
    double pdf(Ptr<Data> dp, bool logscale)const;

    uint state_space_size()const;

    virtual double impute_states();
    virtual void accumulate_sufstats(){}  // should be over-ridden

    void set_initial_state(const Vec &Alpha);
    void set_initial_state_dist(const Vec &mu, const Spd &V);
    void set_initial_state_dist(double mu, double sigsq);     // set all mean to mu, diagonal to sigsq

    virtual void simulate_data(DataSeriesType &, bool initial_state_known=false)const;
    void simulate_state_1(Ptr<StateSpaceData> Old, Ptr<StateSpaceData> New)const;

    virtual ostream & write_state(ostream &, bool nl=true)const;
    virtual ostream & write_smoothed_state(ostream &,bool var=true,bool nl=true)const;

    virtual Vec observed_mean(const Vec &a)const;      // E(y|a) = Z*a
    virtual Spd residual_variance()const=0;            // H_t  p x p
    virtual Spd disturbance_variance()const=0;         // Q_t  r x r

  protected:

    // References to the system matrices are returned to allow
    // polymorphic behavior.  Models with contant dynamics may store
    // the matrices and return references.  Models with time-varying
    // dynamics must store pointers to matrices which are manipulated
    // within the derived model object.

    // State equations can be time dependent, but often are not.  If
    // time dependence is required then a derived class should store a
    // Ptr to an object containing time infomration and set it
    // correctly before calling the kalman recursions.  This will mean
    // over-riding the virtual kalman_filter


    virtual double initialize_first_state(Ptr<ssd>, bool lglk)const;
    virtual void set_T(const Mat &);   // state evolution matrix m x m
    virtual void set_Z(const Mat &);   // E(obs|state) = Z*state p x m
    virtual void set_R(const Mat &);   // m x r
    virtual const Mat & T()const;   // state evolution matrix m x m
    virtual const Mat & Z()const;   // E(obs|state) = Z*state p x m
    virtual const Mat & R()const;   // m x r

    virtual Vec draw_eta()const=0;  // r x 1

    virtual Vec R_eta(const Vec &eta)const; // R*eta m x 1

    virtual Vec T_v(const Vec &a)const;  // returns T(v)
    virtual Mat T_mat(const Mat &P)const;  // returns TP

    virtual Vec Rtrans(const Vec &r)const;   // R^T * r
    virtual Spd ZPZtrans(const Spd &P)const; // ZPZ^T
    virtual Spd TPTtrans(const Mat &P)const; // TPT^T
    virtual Spd RQRtrans()const;             //R Q R^T

  private:
    Mat T_, Z_, R_;
    mutable double last_loglike_;
    mutable bool loglike_set;
    mutable Ptr<ssd> d0;
    mutable Mat L;
    mutable DataSeriesType tmpy;
    mutable Vec a1_;
    mutable Spd P1_;


    // used to simulate alpha.  tmpy is of the same dimensions as regular y
    void setup_tmpy(const DataSeriesType &)const;

    void set_L(const Mat &K, const Mat &Z)const;
    double kalman_update(Ptr<ssd>, const Vec &a, const Spd &P)const;
    double kalman_one(Ptr<ssd> dold, Ptr<ssd> dnew)const;
    void kalman_backward_one(Ptr<ssd> dold, Ptr<ssd> dnew, Mat &)const;
    void dsmooth_one(Ptr<ssd> now, Ptr<ssd> next, bool eta=true, bool eps=false)const;
    void update_r(Ptr<ssd> then, Ptr<ssd> now, const Mat &ell, const Mat &zee)const;

  };
}
#endif // STATE_SPACE_MODEL_HPP
