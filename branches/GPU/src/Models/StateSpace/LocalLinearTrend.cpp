/*
  Copyright (C) 2008 Steven L. Scott

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


#include <Models/StateSpace/LocalLinearTrend.hpp>

namespace BOOM{

typedef LocalLinearTrend LLT;
  LLT::LocalLinearTrend(double sigsq)
      : HomogeneousStateModel(2,1),
        ParamPolicy(new UnivParams(sigsq))
  {
    set_Z(Vec("0 1"));
    set_T(Mat(2,2,Vec("1 1 0 1")));
    set_R(Mat(2,1,Vec("1 1")));
    set_Q(Spd(1, sigsq));
  }

  LLT * LLT::clone()const{return new LLT(*this);}

  Ptr<UnivParams> LLT::Sigsq_prm(){ return ParamPolicy::prm(); }
  const Ptr<UnivParams> LLT::Sigsq_prm()const{ return ParamPolicy::prm(); }
  double LLT::sigsq()const{return Sigsq_prm()->value();}
  void LLT::set_sigsq(double s2){ Sigsq_prm()->set(s2); }

  void LLT::observe_state(const ConstVectorView & now, const ConstVectorView & next){
    suf_.update_raw(next[0] - now[0]); }

  uint LLT::state_size()const{return 2;}
  uint LLT::innovation_size()const{return 1;}

  void LLT::clear_data(){ suf_.clear();}

  const GaussianSuf & LLT::suf(){return suf_;}

  void LLT::combine_data(const Model & mp, bool){
    try{
      const LLT & m(dynamic_cast<const LLT &>(mp));
      suf_.combine(m.suf_);
    }catch(...){
      ostringstream err;
      err << "failed cast in LocalLinearTrend::combine_data" << endl
          << "Model could not be cast to LocalLinearTrend" << endl;
      throw std::runtime_error(err.str());
    }
  }

  void LLT::add_data(Ptr<Data> dp){
    Ptr<VectorData> d(dp.dcast<VectorData>());
    if(!d){
      ostringstream err;
      err << "failed cast in LocalLinearTrend::add_data" << endl
          << "Data could not be cast to VectorData" << endl;
      throw std::runtime_error(err.str());
    }
  }

  double LLT::pdf(Ptr<Data>, bool)const{
    ostringstream err;
    err << "You called the member function 'pdf' for the LocalLinearTrend state model.  "
        << "You probably didn't mean to." << endl;
    throw std::runtime_error(err.str());
    return 0;
  }


}
