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

#ifndef SEASONAL_LOCAL_LEVEL_MODEL_HPP
#define SEASONAL_LOCAL_LEVEL_MODEL_HPP

#include "StateSpaceModel.hpp"
#include <Models/GaussianModel.hpp>

namespace BOOM{

  class SeasonalLocalLevelModel;
  class seasonal_local_level_suf : public sufstat_details<StateSpaceData>
  {
  protected:
    double n_;
    double ssl; // level
    double sss; // seasonal
    double sso; // observed
  public:
    seasonal_local_level_suf();
    seasonal_local_level_suf(const seasonal_local_level_suf &);
    seasonal_local_level_suf * clone()const;

    void Update(const StateSpaceData &d);
    void clear();

    const double & n()const{return n_;}
    const double & sumsq_seasonal()const{return sss;}
    const double & sumsq_level()const{return ssl;}
    const double & sumsq_obs()const{return sso;}
  };
  //======================================================================
  class SeasonalLocalLevelModel
    : public StateSpaceModel,
      public default_param_policy<seasonal_local_level_params>,
      public default_sufstat_policy<seasonal_local_level_suf>,
      public default_prior_policy,
      public loglike_model
  {

    Ptr<GaussianModel> errors;
    Ptr<GaussianModel> seasonal;
    Ptr<GaussianModel> level;

    uint NS_;
    void set_T(uint NS);
    void set_Z(uint NS);
    void set_R(uint NS);
    void set_RQR()const;
    Mat T_;
    Mat Z_;
    Mat R_;
    mutable Spd RQR_;
    mutable bool RQR_is_set;

  public:
    typedef SeasonalLocalLevelModel SLLM;
    typedef data_policy::DataSeriesType DataSeriesType;
    typedef data_policy::data_point_type data_point_type;

    SeasonalLocalLevelModel();
    SeasonalLocalLevelModel(uint NSeasons);
    SeasonalLocalLevelModel(double Siglev, double SigSeason, double SigObs,
			       uint NSeasons);
    SeasonalLocalLevelModel(const std::vector<double> &dv, uint NSeasons);
    SeasonalLocalLevelModel(const SLLM &);
    SLLM * clone()const;

    double loglike(const Vec &Theta)const;
    uint Nseasons()const;

    virtual void draw_params();

    void estimate_initial_state(const std::vector<double> &v);
    void set_data(const DataSeriesType &d);

    double & y(ssd &)const;
    const double & y(const ssd &)const;
    double & level(ssd &)const;
    const double & level(const ssd &)const;
    double & seasonal(ssd &)const;
    const double & seasonal(const ssd &)const;

    double & sigsq_level();
    double & sigsq_seas();
    double & sigsq_obs();
    const double & sigsq_level()const;
    const double & sigsq_seas()const;
    const double & sigsq_obs()const;

    virtual void initialize_params();

    virtual Spd residual_variance(Ptr<ssd>)const;        // H_t
    virtual Spd disturbance_variance(Ptr<ssd>)const;     // Q_t

    virtual Mat & T(const ssd &);                     // state evolution matrix
    virtual Mat & Z(const ssd &);                     // E(obs|state) = Z*state
    virtual Mat & R(const ssd &);
    virtual const Mat & T(const ssd &)const;                     // state evolution matrix
    virtual const Mat & Z(const ssd &)const;                     // E(obs|state) = Z*state
    virtual const Mat & R(const ssd &)const;
    virtual Vec Rtrans(Ptr<ssd>, const Vec &r)const;
    virtual Vec draw_eta(Ptr<ssd>)const;

    virtual Vec observed_mean(const ssd &, const Vec &a)const;         // E(y|a) = Z*a
    virtual Spd dvar_sandwich(Ptr<ssd>)const;            // R Q R^T
    virtual Vec state_mean_disturbance(Ptr<ssd>, const Vec &eta)const;
    // returns R*eta, dimension >= eta;

    virtual Vec state_fwd_vec(Ptr<ssd>, const Vec &a)const;  // returns T(v)
    virtual Mat state_fwd_mat(Ptr<ssd>, const Mat &P)const;  // returns TP
    virtual Spd state_variance_sandwich(Ptr<ssd>, const Spd &P)const; // ZPZ^T

    virtual ostream & write_state(ostream &, bool nl=true)const;
    virtual ostream & write_smoothed_state(ostream &, bool var=true, bool nl=true)const;
  };

  /*------------------------------------------------------------
    The system matrices for this model look like this:
    Assume there are 6 seasons:

    state vector:

        mu_t       today's level
        s_t        today's seasonal effect
	s_t-1      yesterday's seasonal effect
	s_t-2       ..
	s_t-3       ....
	s_t-4      today + 4 days ago gives 5 seasonal number to use for next time


    T = 1  0  0  0  0  0   (today's level centered on yesterday's level)
        0 -1 -1 -1 -1 -1   (-sum of last 5 seasons)
	0  1  0  0  0  0   (move each season back one time period)
        0  0  1  0  0  0    ..
        0  0  0  1  0  0      ..
        0  0  0  0  1  0        ..

    Z = [1 1 0 0 0 0]      Durbin and Watson use a row vector for Z
    Z * alpha = level + seasonal

    R = 1 0    updates only affect today's level and seasonal
        0 1
	0 0
	0 0
	0 0
	0 0
    ------------------------------------------------------------*/


}
#endif //SEASONAL_LOCAL_LEVEL_MODEL_HPP
