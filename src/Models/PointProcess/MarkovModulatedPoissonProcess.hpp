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
/*
#ifndef BOOM_MARKOV_MODULATED_POISSON_PROCESS_HPP_
#define BOOM_MARKOV_MODULATED_POISSON_PROCESS_HPP_

namespace BOOM{

  // The MMPP needs to be a base class, so that the typical MMPP, the
  // Markov-Poisson Cascade, the Marked MMPP, etc. can inherit from it.
  class MarkovModulatedPoissonProcess
      : public IID_DataPolicy<PointProcess>,
        public CompositeParamPolicy,
        public PriorPolicy
  {
   public:
    MarkovModulatedPoissonProcess();
    MarkovModulatedPoissonProcess(const MarkovModulatedPoissonProcess &rhs);
    void MarkovModulatedPoissonProcess * clone()const;

    // Unless the parameters of these processes are known in advance,
    // they should prior distributions and learning methods associated
    // with them before calling simulate_component_posteriors() or
    // sample_posterior();
    void add_process(Ptr<PoissonProcess> birth,
                     Ptr<PoissonProcess> latent,
                     Ptr<PoissonProcess> death);

    virtual MarkovModulatedPoissonProcess * clone()const;

    virtual void sample_posterior();
    virtual double logpri()const;

    virtual double impute_latent_data();
    virtual void simulate_component_posteriors();
    virtual double component_logpri()const;

    // In some MMPP's the hmm state space size grows exponentially
    // with the number of latent processes.
    int hmm_state_space_size()const;
    int number_of_processes()const;
    void impute_latent_data();


    // TODO(stevescot:)
    // Returns the probability that each state was active at time the
    // time of event t.  Rows are states, columns ar times.
    Mat probability_of_activity()const;
    // Returns the probability that each state was responsible for event t
    Mat probability_of_responsibility()const;

    // Returns the integral (from 'then' to 'now') of the sum of the
    // event rates for all the active processes.
    double conditional_cumulative_hazard(const DateTime &then, const DateTime &now,
                             const Selector &activity_state)const;

    // Returns the instantaneous event rate at time 'time' for the
    // process responsible for the transition from 'then' to 'now'.
    // This may be the superposition of several component processes if
    // then==now and several processes are active.
    double conditional_event_rate(int hmm_state then,
                                  int hmm_state now,
                                  const DateTime &time)const;

    PoissonProcess * birth_process(int process_number);
    PoissonProcess * death_process(int process_number);
    PoissonProcess * latent_process(int process_number);

    void finalize_state_specification();

   protected:
    // Declares state r to be a legal state in the model, and
    // associates it with the specified activity pattern.
    void set_activity_state(int r, const Selector &active);


    virtual bool legal_transition(int r, int s) = 0;

    // This function must be called before
    virtual void fill_state_maps();


   private:
    std::vector<Ptr<PoissonProcess> > latent_processes_;
    std::vector<Ptr<PoissonProcess> > birth_processes_;
    std::vector<Ptr<PoissonProcess> > death_processes_;






    // hmm_process_state_ is a mapping from the integers 0,
    // 1,... 2^#processes - 1 into a set of indicators of each process
    // being present/absent.
    std::vector<Selector> process_state_;
    // hmm_state is the reverse mapping, from an activity state to an
    // hmm state in 0..hmm_state_space_size()-1.
    std::map<Selector, int> hmm_state_;

    // Storage needed for forward_backward filtering.
    void resize_filter();
    Vec marginal_state_distribution_;
    Vec one_;
    std::vector<Mat> filter_;

    double last_loglike_;
  };

  class MarkedMMPP : public MarkovModulatedPoissonProcess{
   public:
    MarkedMMPP();
    void add_marked_process(Ptr<PoissonProcess> birth,
                            Ptr<PoissonProcess> latent,
                            Ptr<PoissonProcess> death,
                            Ptr<MixtureComponent> model);
   private:
    std::vector<Ptr<MixtureComponent> > mixture_components_;
  };


}
#endif // BOOM_MARKOV_MODULATED_POISSON_PROCESS_HPP_
*/
