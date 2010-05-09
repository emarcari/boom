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

#ifndef BOOM_CONDITIONAL_MARKOV_MODEL_HPP
#define BOOM_CONDITIONAL_MARKOV_MODEL_HPP

#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/MarkovModel.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/TimeSeries/TimeSeries.hpp>
#include <Models/TimeSeries/AugmentedTimeSeries.hpp>
#include <Models/EmMixtureComponent.hpp>

namespace BOOM{

  class MvnBase;

  class MarkovChoiceData
    : public ChoiceData,
      public MarkovData{
  public:

    // initial data point

    MarkovChoiceData(uint val, uint Nlevels,
		     Ptr<VectorData> subject);
    MarkovChoiceData(const string &lab, Ptr<CatKey>,
		     Ptr<VectorData> subject, bool grow=false);
    MarkovChoiceData(uint val, uint Nlevels,
		     std::vector<Ptr<VectorData> > choice,
		     Ptr<VectorData> subject=0);
    MarkovChoiceData(const string &lab, Ptr<CatKey>,
		     std::vector<Ptr<VectorData> > choice,
		     Ptr<VectorData> subject=0, bool grow=false);

    // subsequent data points
    MarkovChoiceData(uint val, Ptr<MarkovChoiceData> last,
		     Ptr<VectorData> subject);

    MarkovChoiceData(const string &lab, Ptr<MarkovChoiceData> last,
		     Ptr<VectorData> subject, bool grow=false);

    MarkovChoiceData(uint val, Ptr<MarkovChoiceData> last,
		     std::vector<Ptr<VectorData> > choice,
		     Ptr<VectorData> subject=0);
    MarkovChoiceData(const string &lab, Ptr<MarkovChoiceData> last,
		     std::vector<Ptr<VectorData> > choice,
		     Ptr<VectorData> subject=0, bool grow=false);

    MarkovChoiceData(const MarkovChoiceData &rhs, bool copy_links=false);

    virtual MarkovChoiceData * clone()const;
    virtual MarkovChoiceData * create()const;

    virtual uint size(bool=true)const;
    virtual ostream & display(ostream & )const;
    //    virtual istream & read(istream &in);
  };

  //______________________________________________________________________

  typedef AugmentedTimeSeries<MarkovChoiceData, ChoiceData>
  MarkovChoiceDataSeries ;

  class ConditionalMarkovModel
    : public CompositeParamPolicy,
      public TimeSeriesDataPolicy<MarkovChoiceData, MarkovChoiceDataSeries>,
      public PriorPolicy,
      public LoglikeModel
  {
  public:
    typedef MultinomialLogitModel MLM;

    ConditionalMarkovModel(uint S, uint xdim, uint xdim0);
    ConditionalMarkovModel(std::vector<Ptr<MLM> >, Ptr<MLM> pi0);
    ConditionalMarkovModel(const ConditionalMarkovModel &rhs);
    ConditionalMarkovModel * clone()const;

    double pdf(Ptr<Data> dp, bool logscale) const;
    double pdf(Ptr<DataPointType> dp, bool logscale) const;
    double pdf(Ptr<DataSeriesType> dp, bool logscale) const;
    //      double pdf(const DataPointType &dat, bool logscale) const;
    double pdf(const DataSeriesType &dat, bool logscale) const;

    virtual void add_data_point(Ptr<MarkovChoiceData>);
    virtual void clear_data();

    uint state_space_size()const;
    void mle();
    double loglike()const;

    Vec stat_dist(Ptr<ChoiceData>)const;
    Mat Q(Ptr<ChoiceData>)const;
    Mat logQ(Ptr<ChoiceData>)const;
    Vec pi0(Ptr<ChoiceData>)const;

  private:
    std::vector<Ptr<MLM> > mlm_;
    Ptr<MLM> pi0_;
  };

  //______________________________________________________________________
  class CMM_EMC
    : public ConditionalMarkovModel,
      public EmMixtureComponent
  {
  public:
    typedef MultinomialLogitEMC MLME;
    CMM_EMC(std::vector<Ptr<MLME> >, Ptr<MLME> pi0);
    CMM_EMC(const CMM_EMC &rhs);
    CMM_EMC * clone()const;

    // for HMM's
    virtual void add_transition_distribution(Ptr<MarkovChoiceData>, const Mat &P);
    virtual void add_initial_distribution(Ptr<ChoiceData>, const Vec & prob);
    virtual void add_mixture_data(Ptr<Data>, double prob);
    virtual void find_posterior_mode();

    void set_pi0_prior(Ptr<MvnBase>);
    void set_trans_prior(Ptr<MvnBase>); // for one row of transition matrix

  private:
    std::vector<Ptr<MLME> > mlm_;
    Ptr<MLME> pi0_;

    typedef std::vector<Ptr<ChoiceData> > ChoiceVec;

    std::map<Ptr<ChoiceData>, ChoiceVec > pi0_data_;
    std::map<Ptr<MarkovChoiceData>, ChoiceVec > trans_data_;

    Ptr<ChoiceData> get_initial_data_point(Ptr<ChoiceData>, uint s);
    Ptr<ChoiceData> get_transition_data_point(Ptr<MarkovChoiceData>, uint s);
  };


}
#endif // BOOM_CONDITIONAL_MARKOV_MODEL_HPP
