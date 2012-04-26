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
#ifndef BOOM_MLOGIT_BASE_HPP
#define BOOM_MLOGIT_BASE_HPP

#include <Models/Glm/GlmCoefs.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Glm/ChoiceData.hpp>

#include <LinAlg/Array.hpp>

#include <boost/shared_ptr.hpp>

namespace BOOM{

  class MLogitBase
    : public IID_DataPolicy<ChoiceData>,
      public PriorPolicy,
      public NumOptModel,
      public MixtureComponent
  {
  public:
    typedef std::vector<Ptr<CategoricalData> > ResponseVec;

    MLogitBase(uint Nch, uint Psub, uint Pch);
    MLogitBase(ResponseVec responses, const Mat &Xsubject,
	       const Array &Xchoice);
    MLogitBase(ResponseVec responses, const Mat &Xsubject);
    MLogitBase(const std::vector<Ptr<ChoiceData> > &dv);

    MLogitBase(const MLogitBase &rhs);
    MLogitBase * clone()const=0;

    uint beta_size(bool include_zeros=true)const;

    virtual Vec eta(Ptr<ChoiceData>)const=0;
    virtual Vec &fill_eta(const ChoiceData &, Vec &ans)const=0;
    //    virtual Selector inc()const=0;

    virtual void add_all_slopes()=0;
    virtual void drop_all_slopes(bool keep_icpt=true)=0;

    virtual double pdf(Ptr<Data> dp, bool logscale)const;
    virtual double pdf(const Data * dp, bool logscale)const;
    virtual double logp(const ChoiceData & dp)const;
    virtual void add_data(Ptr<ChoiceData>);
    virtual void add_data(Ptr<Data>);

    // simulate an outcome
    uint sim(Ptr<ChoiceData>)const;
    uint sim(Ptr<ChoiceData>, Vec &eta)const;

    // compute all choice probabilities
    Vec predict(Ptr<ChoiceData>)const;  // returns choice probabilities
    Vec & predict(Ptr<ChoiceData>, Vec &ans)const;

    uint subject_nvars()const;
    uint choice_nvars()const;
    uint Nchoices()const;

    void set_sampling_probs(const Vec &probs);
    const Vec & log_sampling_probs()const;
    // probs is an Nchoices vector, where probs[m] gives the
    // probability of keeping in the sample an observation with
    // response level m.  For a prospective study all elements of
    // probs would be 1.

  private:
    mutable boost::shared_ptr<Mat> choice_data_wsp_;
    mutable Vec wsp;
    uint nch_;  // number of choices
    uint psub_; // number of subject X variables
    uint pch_;  // number of choice X variables
    Vec log_sampling_probs_;
  };

}

#endif// BOOM_MLOGIT_BASE_HPP
