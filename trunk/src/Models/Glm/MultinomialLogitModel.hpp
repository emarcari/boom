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

#ifndef BOOM_MULTINOMIAL_LOGIT_MODEL_HPP
#define BOOM_MULTINOMIAL_LOGIT_MODEL_HPP

#include <Models/Glm/MLogitBase.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>

namespace BOOM{

  class MultinomialLogitModel
    : public MLogitBase,
      public ParamPolicy_1<GlmCoefs>
  {
  public:
    typedef std::vector<Ptr<CategoricalData> > ResponseVec;

    // each column of beta_subject corresponds to a different choice.
    MultinomialLogitModel(const Mat & beta_subject, const Vec &beta_choice);

    // the function make_catdat_ptrs can make a ResponseVec out of a
    // vector of strings or uints
    MultinomialLogitModel(const std::vector<uint> &y, const Mat &Xsubject_info,
			  const Array &Xchoice_info);
    MultinomialLogitModel(const std::vector<string> &y, const Mat &Xsubject_info,
			  const Array &Xchoice_info);

    MultinomialLogitModel(const std::vector<uint> & y,    // no choice information
			  const Mat &Xsubject_info);
    MultinomialLogitModel(const std::vector<string> &y,    // no choice information
			  const Mat &Xsubject_info);

    MultinomialLogitModel(ResponseVec responses, const Mat &Xsubject_info,
			  const Array &Xchoice_info); //dim=[#obs, #nch, #pch]
    MultinomialLogitModel(ResponseVec responses,    // no choice information
			  const Mat &Xsubject_info);
    MultinomialLogitModel(uint Nchoices, uint subject_xdim, uint choice_xdim);
    MultinomialLogitModel(const std::vector<Ptr<ChoiceData> > &);
    MultinomialLogitModel(const MultinomialLogitModel &rhs);
    MultinomialLogitModel * clone()const;

    // coefficient vector: elements corresponding to choice level 0
    // (which are constrained to 0 for identifiability) are omitted.
    // Thus beta() is of dimension ((num_choices-1)*psub + pch)

    // If the choices are labelled 0, 1, 2, ..., M-1 then the elements
    // of beta are
    // [ subject_characeristic_beta_for_choice_1,
    //   subject_characeristic_beta_for_choice_2
    //   ...
    //   subject_characeristic_beta_for_choice_M-1
    //   choice_characteristic_beta ]
    const Vec & beta()const;

    // beta_with_zeros() returns the same thing as beta(), but a
    // vector of 0's is prepended, with the zeros corresponding to
    // choice level 0.
    const Vec & beta_with_zeros()const;
    Vec beta_subject(uint choice)const;
    Vec beta_choice()const;

    void set_beta(const Vec &b, bool reset_inc=false);
    void set_beta_subject(const Vec &b, uint i);
    void set_beta_choice(const Vec &b);

    Ptr<GlmCoefs> coef();
    const Ptr<GlmCoefs> coef()const;
    Selector inc()const;

    virtual double Loglike(Vec &g, Mat &H, uint nd)const;
    virtual void drop_all_slopes(bool keep_intercept=true);
    virtual void add_all_slopes();

//     // compute beta^Tx for the choice and subject portions of X
     double predict_choice(Ptr<ChoiceData>, uint m)const;
     double predict_subject(Ptr<ChoiceData>, uint m)const;

    // computes all logits
    virtual Vec eta(Ptr<ChoiceData>)const;
    virtual Vec &fill_eta(const ChoiceData &, Vec &ans)const;

  private:
    mutable Vec beta_with_zeros_;
    mutable bool beta_with_zeros_current_;

    void watch_beta();
    void setup();
    void setup_observers();
    void fill_extended_beta()const;
    void index_out_of_bounds(uint m)const;
  };

  //______________________________________________________________________

  class MvnBase;
  class MultinomialLogitEMC  // EMC = EmMixtureComponent
    : public MultinomialLogitModel,
      public EmMixtureComponent
  {
  public:
    MultinomialLogitEMC(const Mat & beta_subject, const Vec &beta_choice);
    MultinomialLogitEMC(uint Nchoices, uint subject_xdim, uint choice_xdim);
    MultinomialLogitEMC * clone()const;

    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
    virtual double pdf(Ptr<Data> dp, bool logscale)const{
      return MultinomialLogitModel::pdf(dp, logscale);}
    virtual double pdf(const Data * dp, bool logscale)const{
      return MultinomialLogitModel::pdf(dp, logscale);}

    void add_mixture_data(Ptr<Data>, double prob);
    void clear_data();

    virtual void find_posterior_mode();
    void set_prior(Ptr<MvnBase>);
    // assumes a posterior sampler derived from MLVS_base
  private:
    Vec probs_;
    Ptr<MvnBase> pri_;
  };
}
#endif// BOOM_MULTINOMIAL_LOGIT_MODEL_HPP
