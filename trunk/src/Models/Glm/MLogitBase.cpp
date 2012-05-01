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
  MLB::MLogitBase(ResponseVec responses,
                  const Mat &Xsubject,
		  const std::vector<Mat> &Xchoice)
    : DataPolicy(),
      PriorPolicy(),
      choice_data_wsp_(new Mat),
      nch_(responses[0]->nlevels()),
      psub_(Xsubject.ncol()),
      pch_(0)
  {
    uint n = responses.size();
    if(nrow(Xsubject) != n
       || (!Xchoice.empty() && Xchoice.size() != n)){
      ostringstream err;
      err << "Predictor sizes do not match in MLogitBase constructor" << endl
          << "responses.size() = " << n << endl
          << "nrow(Xsubject)   = " << nrow(Xsubject) << endl;
      if(!Xchoice.empty()){
        err << "Xchoice.size()   = " << Xchoice.size() << endl;
      }
      report_error(err.str());
    }

    for(uint i=0; i<n; ++i){
      NEW(VectorData, subject)(Xsubject.row(i));
      std::vector<Ptr<VectorData> > choice;
      if(!Xchoice.empty()){
        const Mat & choice_matrix(Xchoice[i]);
        if(pch_ == 0){
          pch_ = ncol(choice_matrix);
        }else if(pch_ != ncol(choice_matrix)){
          ostringstream err;
          err << "The number of columns in the choice matrix for observation "
              << i
              << " did not match previous observations." << endl
              << "ncol(Xsubject[i]) = " << ncol(choice_matrix) << endl
              << "previously:         " << pch_ << endl;
          report_error(err.str());
        }

        if(nrow(choice_matrix) != nch_){
          ostringstream err;
          err << "The number of rows in choice matrix does not match the "
              << "number of choices available in the response." << endl
              << "response:  " << nch_ << endl
              << "Xchoice[" << i << "]: " << nrow(choice_matrix) << endl;
          report_error(err.str());
        }
        for(uint j=0; j<nch_; ++j){
          NEW(VectorData, ch)(choice_matrix.row(j));
          choice.push_back(ch);
        }
      }

      NEW(ChoiceData, dp)(*responses[i], subject, choice);
      add_data(dp);
    }

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
    DataPolicy::add_data(dp);
  }

  void MLB::add_data(Ptr<Data> d){
    Ptr<ChoiceData> dp = d.dcast<ChoiceData>();
    this->add_data(dp);
  }

}
