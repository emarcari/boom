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

#ifndef BOOM_HEALTH_STATE_MODEL_HPP_
#define BOOM_HEALTH_STATE_MODEL_HPP_

#include <Models/DataTypes.hpp>
#include <Models/TimeSeries/TimeSeriesDataPolicy.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/MarkovModel.hpp>

namespace BOOM{

  class HealthStateData : public Data {
   public:
    HealthStateData(Ptr<Data>, int treatment);
    HealthStateData(const HealthStateData &rhs);

    // Required virtual functions...
    virtual HealthStateData * clone()const;
    virtual ostream & display(ostream &out)const;

    // If the subject switched treatment groups during the time period
    // covered by this data point, then split_treatment can be used to
    // specify the treatment group the subject started in, and the
    // fraction of time spent in the final treatment.  It is assumed
    // that at most one treatment switch is possible, so the fraction
    // of time in initial_treatment is 1-final_treatment_fraction.
    void split_treatment(int initial_treatment, double final_treatment_fraction);

    // Treatment group to which the subject belonged at the end of the time period.
    int treatment()const;
    // Fraction of the preceding time period that the subject belonged
    // to the treatment group indicated by treatment().
    double final_treatment_fraction()const;

    // Treatment group to which the subject belonged at the start of
    // the time period.  This will
    int initial_treatment()const;

    Ptr<Data> shared_value();
    const Data * value()const;

   private:
    Ptr<Data> value_;

    // The treatment level this subject was in at the end of the time
    // period covered by this data point.
    int treatment_;

    // The treatment level this subject was in at the beginning of the
    // time period covered by this data point.
    int initial_treatment_;

    // Fraction of the time period covered by this data point that the
    // subject was in the treatment group described by treatment_.
    double final_treatment_fraction_;
  };
  //======================================================================
  class HealthStateModel :
      public TimeSeriesDataPolicy<HealthStateData>,
      public CompositeParamPolicy,
      public PriorPolicy,
      public LoglikeModel
  {
   public:
    HealthStateModel(const std::vector<Ptr<MixtureComponent> > &mix,
                     const std::vector<Ptr<MarkovModel> > &mark);

    HealthStateModel(const HealthStateModel &rhs);
    virtual HealthStateModel * clone()const;

    virtual double loglike()const;

    uint state_space_size()const;
    uint ntreatments()const;
    void clear_client_data();

    std::vector<Ptr<MixtureComponent> > mixture_components();
    Ptr<MixtureComponent> mixture_component(uint s);

    double impute_latent_data();
    Ptr<MarkovModel> mark(int treatment);
   private:
    // Add mix_ and mark_ to the list of models managed by the
    // CompositeParamPolicy.
    void initialize_param_policy();

    double initialize_fwd(Ptr<HealthStateData>)const;
    double fwd(const TimeSeries<HealthStateData> &series);
    double compute_loglike(const TimeSeries<HealthStateData> &series)const;
    void bkwd(const TimeSeries<HealthStateData> &series);

    // Fill logp_[0..state_space_size()-1] with the conditional
    // probability density of the given data point under each mixture
    // component.
    void fill_logp(Ptr<HealthStateData>);
    void fill_logp(Ptr<HealthStateData>, Vec &logp)const;

    void fill_logQ(Ptr<HealthStateData>);
    void fill_logQ(Ptr<HealthStateData>, Mat &logQ)const;

    int sample_treatment(Ptr<HealthStateData>,
                         uint previous_state,
                         uint current_state);

    std::vector<Ptr<MixtureComponent> > mix_;
    std::vector<Ptr<MarkovModel> > mark_;

    std::map<Ptr<HealthStateData>, Vec> state_prob_hist_;
    double loglike_;
    double logpost_;
    mutable Vec logp_;
    mutable Mat logQ_;

    std::vector<Mat> P_;  // joint distribution of state in FB
    mutable Vec pi_;    // marginal distribution of state in FB
    Vec one_;   // vector of 1's of dimension state_space_size
  };

}

#endif //  BOOM_HEALTH_STATE_MODEL_HPP_
