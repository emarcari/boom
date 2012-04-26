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

#ifndef BOOM_MULTINOMIAL_LOGIT_MODEL_SPLIT_HPP
#define BOOM_MULTINOMIAL_LOGIT_MODEL_SPLIT_HPP

#include <Models/Glm/MLogitBase.hpp>
#include <Models/Policies/ManyParamPolicy.hpp>

namespace BOOM{

  class MLogitSplit
    : public MLogitBase,
      public ManyParamPolicy
  {
  public:

    // each column of beta_subject corresponds to a different choice.
    MLogitSplit(const Mat & beta_subject, const Vec &beta_choice);

    // the function make_catdat_ptrs can make a ResponseVec out of a
    // vector of strings or uints
    MLogitSplit(ResponseVec responses, const Mat &Xsubject_info,
		const Array &Xchoice_info);
    // dim(Xchoice_info) = [#obs, #choices, #choice x's]

    MLogitSplit(ResponseVec responses,    // no choice information
		const Mat &Xsubject_info);

    MLogitSplit(const std::vector<Ptr<ChoiceData> > &);
    MLogitSplit(uint Nchoices, uint Xdim_subject, uint Xdim_choice=0);
    MLogitSplit(const MLogitSplit &rhs);
    MLogitSplit * clone()const;

    virtual Vec eta(Ptr<ChoiceData>)const;
    virtual Vec &fill_eta(const ChoiceData &, Vec &ans)const;

    virtual double Loglike(Vec &g, Mat &h, uint nd)const;

    Vec beta_subject(uint choice)const;
    Vec beta_choice()const;
    void set_beta_subject(const Vec &b, uint m, bool infer_inc=false); // m in [1,nch)
    void set_inc_subject(const Selector &inc, uint m);
    void set_beta_choice(const Vec &b, bool infer_inc=false);
    void set_inc_choice(const Selector &inc);

    virtual void drop_all_slopes(bool keep_intercept=true);
    virtual void add_all_slopes();

    Ptr<GlmCoefs> Beta_choice_prm();
    Ptr<GlmCoefs> Beta_subject_prm(uint i);
    const Ptr<GlmCoefs> Beta_choice_prm()const;
    const Ptr<GlmCoefs> Beta_subject_prm(uint i)const;

    // compute beta^Tx for the choice and subject portions of X
    double predict_choice(const ChoiceData &, uint m)const;
    double predict_subject(const ChoiceData &, uint m)const;

  private:
    std::vector<Ptr<GlmCoefs> >beta_subject_; // no leading vector of 0's
    Ptr<GlmCoefs> beta_choice_;

    void setup_params();
    void setup_beta_subject();

    typedef std::vector<Vec> VV;
    void update_grad(const Vec & probs, Ptr<ChoiceData>, Vec &g,
		     Vec & xbar, VV & xx, VV & ww)const;
    void update_hess(const Vec & probs, const Vec &xbar, Mat &h,
		     const VV & xx, const VV & ww)const;
  };


}
#endif// BOOM_MULTINOMIAL_LOGIT_MODEL_SPLIT_HPP
