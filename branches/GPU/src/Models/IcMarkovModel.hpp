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

#ifndef BOOM_IC_MARKOV_MODEL_HPP_
#define BOOM_IC_MARKOV_MODEL_HPP_

#include <Models/MarkovModel.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{

  class IcMarkovModel
    : public ParamPolicy_2<TransitionProbabilityMatrix,GlmCoefs>,
      public TimeSeriesSufstatDataPolicy<MarkovData, MarkovDataSeries, MarkovSuf>,
      public ConjugatePriorPolicy<MarkovConjSampler>,
      public LoglikeModel
  {
  public:
    IcMarkovModel(uint S, uint dim);
    IcMarkovModel(const Mat & Q, const Vec & beta);
    IcMarkovModel * clone()const;

    double pdf(Ptr<Data> dp, bool logscale) const;
    double pdf(Ptr<DataPointType> dp, bool logscale) const;
    double pdf(Ptr<DataSeriesType> dp, bool logscale) const;
    double pdf(const DataPointType &dat, bool logscale) const;
    double pdf(const DataSeriesType &dat, bool logscale) const;

    Ptr<TPM> Q_prm();
    const Ptr<TPM> Q_prm()const;
    virtual const Mat &Q()const;
    virtual void set_Q(const Mat &Q)const;
    double Q(uint, uint)const;

    Vec pi0(const Vec &x)const;
    Ptr<GlmCoefs> coef();
    const Ptr<GlmCoefs> coef()const;
    Selector inc()const;

    double loglike()const;
    Vec stat_dist()const;

  private:
    Ptr<MultinomialLogitModel> pi0_;
    Ptr<MarkovData> prototype_;
  }
}
#endif  // BOOM_IC_MARKOV_MODEL_HPP_
