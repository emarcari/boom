/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_POISSON_CLUSTER_PROCESS_HPP_
#define BOOM_POISSON_CLUSTER_PROCESS_HPP_

#include <Models/PointProcess/PointProcess.hpp>
#include <Models/PointProcess/PoissonProcess.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

#include <LinAlg/Selector.hpp>
#include <vector>
#include <map>

namespace BOOM{

  // A Poisson cluster process is a type of Markov modulated Poisson
  // process.  There is a baseline process that sweeps up stray
  // events.  When the primary Poisson process generates a top-level
  // event, it also activates a secondary process that generates
  // events subsequent events until it dies.
  class PoissonClusterProcess
      : public CompositeParamPolicy,
        public IID_DataPolicy<PointProcess>,
        public PriorPolicy
  {
   public:
    // Use this constructor if there are no marks in the process, or
    // if you don't want to model the marks.
    PoissonClusterProcess(Ptr<PoissonProcess> background,
                          Ptr<PoissonProcess> primary_activity_birth,
                          Ptr<PoissonProcess> primary_traffic,
                          Ptr<PoissonProcess> primary_activity_death,
                          Ptr<PoissonProcess> secondary_traffic,
                          Ptr<PoissonProcess> secondary_activity_death);

    // Use this constructor if there are marks to be modeled.
    PoissonClusterProcess(Ptr<PoissonProcess> background,
                          Ptr<PoissonProcess> primary_birth,
                          Ptr<PoissonProcess> primary_traffic,
                          Ptr<PoissonProcess> primary_death,
                          Ptr<PoissonProcess> secondary_traffic,
                          Ptr<PoissonProcess> secondary_death,
                          Ptr<MixtureComponent> primary_mark_model,
                          Ptr<MixtureComponent> secondary_mark_model);

    PoissonClusterProcess(const PoissonClusterProcess &rhs);
    PoissonClusterProcess * clone()const;

    virtual ~PoissonClusterProcess(){}

    void set_mark_models(Ptr<MixtureComponent> primary,
                         Ptr<MixtureComponent> secondary);

    virtual void clear_client_data();
    void impute_latent_data(RNG &rng);

    // Sample the posterior distributions of the client models.  To be
    // called after impute_latent_data().
    virtual void sample_client_posterior();
    virtual double logpri()const;

    // The log-event rate at time t for the process (or superposition
    // of processes) responsible for a transition from hmm state r to
    // hmm state s.  If r==s then this is the sum of the event rates
    // for processes active in state r.  Otherwise it is the rate of
    // the birth or death process associated with the process
    // responsible for the change in the activity state.
    virtual double conditional_event_loglikelihood(
        int r, int s, const PointProcessEvent &event,
        double logp_primary, double logp_secondary, int source)const;

    // The sum of cumulative hazard functions between times t0 and t1
    // for processes active at time t0.  The set of active processes
    // is determined by hmm state 'r', using the function
    // active_processes(r).
    double conditional_cumulative_hazard(
        const DateTime &t0, const DateTime &t1, int r)const;

    int number_of_hmm_states()const;

    double filter(const PointProcess &data, const std::vector<int> &source);
    double initialize_filter(const PointProcess &data);

    // Fills position t in the filter_ member with the conditional
    // distribution of activity state (r, s) given observed data up to
    // time t.  Returns the conditional log likelihood of observation
    // t given preceding observations.  If source > 0 then the update
    // is done conditional on knowledge that latent process 'source'
    // (or its associated birth or death processes) produced the event
    // at time 't'.
    double fwd_1(const PointProcess &data, int t, int source);

    // Backward sampling simulates the activity state of the process
    // at each point in time.  Along the way it ascribes each event to
    // one of the latent processes, or the associated birth and death
    // processes.  The 'source' argument can be an empty vector, in
    // which case unsupervised learning takes place (the normal case
    // for hmm's).  Otherwise the backward sampling is conditional on
    // the knowledge of which latent process produced the event.
    // There is still sampling to do because (a) the source for some
    // events may be unknown (represented by a -1 element in
    // 'source'), and (b) it must be determined whether the event was
    // produced by the specified latent process or by the associated
    // birth or death process.
    void backward_sampling(RNG &rng,
                           const PointProcess &data,
                           const std::vector<int> &source,
                           Mat &probability_of_activity,
                           Mat &probability_of_responsibility);

    int draw_previous_state(RNG &rng, int time, int current_state);

    // Determine the specific process responsible for the event at
    // time t, given that the state at time t-1 is prev_state and the
    // state at time t is current_state.  If the states are the same
    // then a further Monte Carlo draw is used to determine the
    // responsible process.
    virtual PoissonProcess * assign_responsibility(
        RNG &rng, const PointProcess &data, int t,
        int previous_state, int current_state, int source);

    // Attribute the event at time 'current_time' to the responsible
    // process and update its sufficient statistics accordingly,
    // including the sufficient statistics of the associated mark
    // models.  Exposure time is not updated, because it has already
    // been updated with update_exposure_time.
    virtual void attribute_event(const PointProcessEvent &data,
                                 PoissonProcess* responsible_process);

    // Update the statistics for all the processes determined to be
    // running between current_time and current_time + 1, conditional
    // on the values of current_state and next_state.
    void update_exposure_time(const PointProcess &data, int current_time,
                              int previous_state, int current_state);

    double loglike()const{return last_loglike_;}

    virtual void clear_data();
    virtual void add_data(Ptr<Data> dp);  // *dp is a PointProcess
    virtual void add_data(Ptr<PointProcess> dp);
    void add_supervised_data(Ptr<PointProcess> dp,
                             const std::vector<int> &source);

    // Simulate a PoissonClusterProcess observed from t0 to t1.
    virtual PointProcess simulate(const DateTime &t0, const DateTime &t1)const;

    const std::vector<Mat> & probability_of_activity()const;
    const std::vector<Mat> & probability_of_responsibility()const;

    void record_activity(VectorView activity_probs, int state);
    void record_responsibility(VectorView activity_probs,
                               PoissonProcess* responsible_process);

    // Indicates whether 'responsible_process' could have produced an
    // event if the hmm state was a transition from previous_hmm_state
    // to current_state.
    bool allows_production(int previous_hmm_state,
                           int current_hmm_state,
                           int responsible_process)const;

    // These functions can return 0/NULL if no mark_models have been
    // assigned.
    MixtureComponent * mark_model(const PoissonProcess * process);
    const MixtureComponent * mark_model(const PoissonProcess * process)const;

   private:
    void initialize();
    void fill_state_maps();  // make virtual
    void setup_filter();
    virtual void register_models_with_param_policy();

    // Returns true if process is associated with a primary event.
    // I.e. primary_traffic, primary_birth, or primary_death.
    bool primary(const PoissonProcess *process)const;

    std::vector<PoissonProcess *> get_responsible_processes(
        int r, int s, int source);
    std::vector<const PoissonProcess *> get_responsible_processes(
        int r, int s, int source)const;

    std::vector<PoissonProcess *> subset_matching_source(
        std::vector<PoissonProcess *> &, int source);
    std::vector<const PoissonProcess *> subset_matching_source(
        const std::vector<PoissonProcess *> &, int source)const;

    // Returns true if process is associated with latent process
    // 'source', where source is 0, 1, or 2.
    bool matches_source(const PoissonProcess *process, int source)const;

    Ptr<PoissonProcess> background_;

    Ptr<PoissonProcess> primary_birth_;
    Ptr<PoissonProcess> primary_death_;
    Ptr<PoissonProcess> primary_traffic_;

    Ptr<PoissonProcess> secondary_traffic_;
    Ptr<PoissonProcess> secondary_death_;

    // Responsible for user births, traffic, and deaths
    Ptr<MixtureComponent> primary_mark_model_;

    // Responsible for machine deaths, and machine traffic, including
    // background traffic.
    Ptr<MixtureComponent> secondary_mark_model_;

    //  Indicates which processes are on/off in each hmm state
    std::vector<Selector> activity_state_;

    // Holds the set of legal given the current state, for use in
    // fwd_1.
    std::vector<std::vector<int> > legal_target_transitions_;

    // Holds the vector of processes active in each HMM
    // state, including birth and death processes.
    std::vector<std::vector<PoissonProcess *> > active_processes_;

    // Keeps track of which processes are potentially responsible for
    // an (r->s) transition.  If a transition is impossible then no
    // map entry will be present.
    typedef
    std::map<std::pair<int, int>, std::vector<PoissonProcess *> >
    ResponsibleProcessMap;
    ResponsibleProcessMap responsible_process_map_;

    std::vector<Mat> filter_;
    Vec pi0_;
    mutable Vec wsp_;
    Vec one_;
    double last_loglike_;

    // Each vector element corresponds to the PointProcess for a
    // single subject.  Space for a new subject is allocated when
    // add_data is called.  Each matrix has a number of rows equal to
    // the number of latent processes, and a number of columns equal
    // to the number of events in that subjects PointProcess data.
    std::vector<Mat> probability_of_activity_;
    std::vector<Mat> probability_of_responsibility_;

    enum InitializationStrategy{
      UniformInitialState = 0, StationaryDistribution};
    InitializationStrategy initialization_strategy_;

    // The known_source_store_ keeps track of source information for
    // each PointProcess.  If some events are known to be
    typedef std::map<Ptr<PointProcess>, std::vector<int> > SourceMap;
    SourceMap known_source_store_;

  };

}

#endif// BOOM_POISSON_CLUSTER_PROCESS_HPP_
