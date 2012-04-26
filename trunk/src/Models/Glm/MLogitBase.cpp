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
#include "MLogitBase.hpp"
#include <cmath>

#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/lse.hpp>

#include <LinAlg/Array3.hpp>
#include <numopt.hpp>

#include <boost/bind.hpp>

namespace BOOM{

  typedef MLogitBase MLB;

  MLB::MLogitBase(uint Nch, uint Psub, uint Pch)
    : DataPolicy(),
      PriorPolicy(),
      choice_data_wsp_(new Mat),
      nch_(Nch),
      psub_(Psub),
      pch_(Pch)
  {
  }
  //------------------------------------------------------------
  MLB::MLogitBase(ResponseVec responses, const Mat &Xsubject,
		  const Array &Xchoice)
    : DataPolicy(),
      PriorPolicy(),
      choice_data_wsp_(new Mat),
      nch_(responses[0]->nlevels()),
      psub_(Xsubject.ncol()),
      pch_(Xchoice.dim(2))
  {
    assert(nch_==Xchoice.dim(1));

    uint n = responses.size();
    Ptr<CatKey> key = responses[0]->key();
    for(uint i=0; i<n; ++i){
      NEW(VectorData, subject)(Xsubject.row(i));
      std::vector<Ptr<VectorData> > choice;
      for(uint j=0; j<nch_; ++j){
	NEW(VectorData, ch)(Xchoice.vector_slice(Array::index3(i, j, -1)));
	choice.push_back(ch);
      }

      NEW(ChoiceData, dp)(responses[i]->lab(), key, choice, subject);
      dp->set_wsp(choice_data_wsp_);
      add_data(dp);
    }

  }
  //------------------------------------------------------------
  MLB::MLogitBase(ResponseVec responses, const Mat &Xsubject)
    : DataPolicy(),
      PriorPolicy(),
      choice_data_wsp_(new Mat),
      nch_(responses[0]->nlevels()),
      psub_(Xsubject.ncol()),
      pch_(0)
  {
    uint n = responses.size();
    Ptr<CatKey> key = responses[0]->key();
    for(uint i=0; i<n; ++i){
      NEW(VectorData, xsub)(Xsubject.row(i));
      Ptr<ChoiceData> dp(new ChoiceData(responses[i]->lab(), key, xsub));
      dp->set_wsp(choice_data_wsp_);
      add_data(dp);
    }
  }

  MLB::MLogitBase(const std::vector<Ptr<ChoiceData> >  &dv)
    : DataPolicy(),
      PriorPolicy(),
      choice_data_wsp_(new Mat),
      nch_(dv[0]->nchoices()),
      psub_(dv[0]->subject_nvars()),
      pch_(dv[0]->choice_nvars())
  {
    set_data(dv.begin(), dv.end());
  }

  MLB::MLogitBase(const MLogitBase &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      NumOptModel(rhs),
      choice_data_wsp_(new Mat(*rhs.choice_data_wsp_)),
      nch_(rhs.nch_),
      psub_(rhs.psub_),
      pch_(rhs.pch_)
  {
  }

  //------------------------------------------------------------
  double MLB::logp(const ChoiceData & dp)const{
    // for right now...  assumes all choices are available to everyone
    //    uint n = dp->n_avail();
    wsp.resize(nch_);
    fill_eta(dp, wsp);
    uint y = dp.value();
    double ans = wsp[y] - lse(wsp);
    return ans;
  }
  //------------------------------------------------------------
  double MLB::pdf(Ptr<Data> dp, bool logscale)const{
    double ans = logp(*DAT(dp));
    return logscale ? ans : exp(ans);
  }

  double MLB::pdf(const Data * dp, bool logscale)const{
    double ans = logp(*DAT(dp));
    return logscale ? ans : exp(ans);
  }

  //------------------------------------------------------------


  uint MLB::beta_size(bool include_zeros)const{
    uint nch(nch_);
    if(!include_zeros)  --nch;
    return nch*psub_ + pch_;
  }

  uint MLB::sim(Ptr<ChoiceData> dp, Vec &prob)const{
    predict(dp, prob);
    return rmulti(prob);
  }

  uint MLB::sim(Ptr<ChoiceData> dp)const{
    Vec prob = predict(dp);
    return rmulti(prob);
  }

  Vec & MLB::predict(Ptr<ChoiceData> dp, Vec &ans)const{
    fill_eta(*dp, ans);
    ans = exp(ans-lse(ans));
    return ans;
  }

  Vec MLB::predict(Ptr<ChoiceData> dp)const{
    Vec ans(nch_);
    return predict(dp,ans);
  }

  //------------------------------------------------------------
  uint MLB::subject_nvars()const{ return psub_;}
  uint MLB::choice_nvars()const{return pch_;}
  uint MLB::Nchoices()const{return nch_;}
  //------------------------------------------------------------

  void MLB::set_sampling_probs(const Vec & probs){
    assert(probs.size()==nch_);
    log_sampling_probs_ = log(probs);
  }

  const Vec & MLB::log_sampling_probs()const{
    return log_sampling_probs_;}

  //------------------------------------------------------------


  void MLB::add_data(Ptr<ChoiceData> dp){
    dp->set_wsp(choice_data_wsp_);
    DataPolicy::add_data(dp);
  }

  void MLB::add_data(Ptr<Data> d){
    Ptr<ChoiceData> dp = d.dcast<ChoiceData>();
    this->add_data(dp);
  }

}
