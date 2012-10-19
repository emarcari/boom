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

#ifndef BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_
#define BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/PoissonRegressionModel.hpp>
#include <Models/Glm/WeightedRegressionModel.hpp>
#include <Models/MvnBase.hpp>
#include <Models/Glm/PosteriorSamplers/NormalMixtureApproximation.hpp>

namespace BOOM{

class PoissonRegressionAuxMixSampler : public PosteriorSampler {
 public:
  PoissonRegressionAuxMixSampler(PoissonRegressionModel *model,
                                 Ptr<MvnBase> prior,
                                 int number_of_threads = 1);

  virtual void draw();
  virtual double logpri()const;

  // Below this line are implementation details exposed for testing.
  void impute_latent_data();
  void impute_latent_data_single_threaded();
  double draw_final_event_time(int y);
  double draw_censored_event_time(double final_event_time, double rate);
  double draw_censored_event_time_zero_case(double rate);

  // Find the parameters of the Gaussian mixture responsible for this
  // value of u = -log(tau) - eta.
  void unmix(double u, int y, double *mu, double *sigsq);

  void draw_beta_given_complete_data();
  const WeightedRegSuf &complete_data_sufficient_statistics()const;

 private:
  // This private constructor is called by the public constructor to
  // populate data_imputers_ if number_of_threads > 1 in the public
  // constructor.
  PoissonRegressionAuxMixSampler(PoissonRegressionModel *model,
                                 Ptr<MvnBase> prior,
                                 int number_of_threads,
                                 int thread_id);

  PoissonRegressionModel *model_;
  Ptr<MvnBase> prior_;
  WeightedRegSuf complete_data_suf_;

  // A flag when running in 'master mode'
  bool first_time_;

  // The NormalMixtureApproximationTable is really big.  We'll make it
  // static so that multiple samplers (e.g. in a hierarchical model)
  // don't all need their own copy.  Note that during the first MCMC
  // iteration the table can get modified, so we need to run in single
  // threaded mode for a single iteration.  After all possible values
  // of y have been observed it is safe to run in multi-threaded mode.
  static NormalMixtureApproximationTable mixture_table_;

  // num_threads_ and thread_id_ are used by slaves in the impute_data
  // method.  To check the number of threads in the master, you should
  // call data_imputers_.size(), and not rely on num_threads_.
  int num_threads_;
  int thread_id_;
  typedef boost::shared_ptr<PoissonRegressionAuxMixSampler> DataImputerPtr;
  std::vector<DataImputerPtr> data_imputers_;
};

}  // namespace BOOM

#endif // BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_
