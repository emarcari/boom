/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_HIDDEN_CONDITIONAL_MARKOV_MODEL_HPP
#define BOOM_HIDDEN_CONDITIONAL_MARKOV_MODEL_HPP
#include <BOOM.hpp>
#include <vector>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/TimeSeries/TimeSeries.hpp>
#include <Models/TimeSeries/AugmentedTimeSeriesDataPolicy.hpp>
#include <Models/TimeSeries/AugmentedTimeSeries.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/DataTypes.hpp>
#include <Models/DataPair.hpp>
#include <Models/ConditionalMarkovModel.hpp>

namespace BOOM{
  class EmMixtureComponent;

  typedef DataPair<Data, MarkovChoiceData> ChmmData;
  typedef AugmentedTimeSeries<ChmmData, ChoiceData> ChmmSeries;
    class ConditionalHmmFilter;
    class ConditionalHmmEmFilter;
    class ConditionalHmmGibbsFilter;

    class HiddenConditionalMarkovModel
      : public AugmentedTimeSeriesDataPolicy<ChmmData, ChoiceData>,
	public CompositeParamPolicy,
	public PriorPolicy,
	public LoglikeModel
    {
    public:
      typedef std::vector<Ptr<Model> > ModelVec;
      typedef ModelVec::size_type sz;
      typedef ConditionalMarkovModel CMM;

      HiddenConditionalMarkovModel(ModelVec Mix, Ptr<CMM> Mark);
      HiddenConditionalMarkovModel(const HiddenConditionalMarkovModel &rhs);
      virtual HiddenConditionalMarkovModel * clone()const;

      uint state_space_size() const;
      virtual void initialize_params();

      double pdf(dPtr dp, bool logscale) const;
      virtual void clear_client_data();
      void clear_prob_hist();

      virtual double loglike()const;
      void randomly_assign_data();

      void save_loglike(const string & fname, uint ping=1);
      // where to save it, buffer update freq.

      void write_loglike(double)const;
      // write the number

    protected:
      template <class Fwd>        // needed for derived copy constructor
      void set_mixture_components(Fwd b, Fwd e){
	mix_.assign(b,e);
	ParamPolicy::set_models(b,e);
	ParamPolicy::add_model(mark_);
      }
      void set_filter(boost::shared_ptr<ConditionalHmmFilter> f);
    private:
      std::vector<Ptr<Model> > mix_;
      Ptr<CMM> mark_;
      Ptr<UnivParams> loglike_;
      boost::shared_ptr<ConditionalHmmFilter> filter_;
      std::map<Ptr<Data>, Vec > prob_hist_;

      void setup();
    };

    //______________________________________________________________________

    class HCMM_EM
      : public HiddenConditionalMarkovModel
    {
    public:
      HCMM_EM(std::vector<Ptr<EmMixtureComponent> > Mix,
	      Ptr<CMM_EMC> Mark);

      HCMM_EM(const HCMM_EM &rhs);
      HCMM_EM * clone()const;

      virtual void initialize_params();
      virtual void mle();
      void find_posterior_mode();

      double Estep(bool bayes=false);
      void Mstep(bool bayes=false);
      void trace(bool=true);
      void set_epsilon(double);

    private:
      void find_mode(bool bayes=false, bool save_history=false);
      void setup();

      std::vector<Ptr<EmMixtureComponent> > mix_;
      Ptr<CMM_EMC> mark_;
      boost::shared_ptr<ConditionalHmmEmFilter> filter_;
      double eps;
      bool trace_;
    };
}
#endif// BOOM_HIDDEN_CONDITIONAL_MARKOV_MODEL_HPP
