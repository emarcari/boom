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
#include "ChoiceData.hpp"
#include <LinAlg/Vector.hpp>

namespace BOOM{

  typedef ChoiceData CHD;
  typedef CategoricalData CAT;
  typedef VectorData VD;

//   CHD::ChoiceData(Ptr<CAT> y, const Vec & subject, const Mat &choice,
// 		  bool add_subject_icpt, bool add_choice_icpt)
//     : y_(y),
//       xsubject_(),
//       xchoice_(choice.nrow())
//   {
//     if(add_subject_icpt) xsubject_ = new VD(concat(1.0, subject));
//     else xsubject_ = new VD(subject);

//     uint n = choice.nrow();
//     assert(n==0 || n==y->nlevels());
//     for(uint i=0; i<n; ++i){
//       if(add_choice_icpt) xchoice_[i] = new VD(concat(1.0, choice.row(i)));
//       else xchoice_[i] = new VD(choice.row(i));
//     }
//   }

  CHD::ChoiceData(uint val, uint Nlevels, Ptr<VectorData> subject)
    : CAT(val, Nlevels),
      xsubject_(subject),
      xchoice_()
  {}


  CHD::ChoiceData(uint val, Ptr<CatKey> key, Ptr<VectorData> subject)
    : CAT(val, key),
      xsubject_(subject),
      xchoice_()
  {}

  CHD::ChoiceData(const string & lab, Ptr<CatKey> key,
	       Ptr<VectorData> subject, bool grow)
    : CAT(lab, key, grow),
      xsubject_(subject),
      xchoice_()
  {}

  CHD::ChoiceData(uint val, Ptr<ChoiceData> last,
	       Ptr<VectorData> subject)
    : CAT(val, last->key()),
      xsubject_(subject),
      xchoice_()
  {}

  CHD::ChoiceData(const string & lab, Ptr<ChoiceData> last,
	       Ptr<VectorData> subject, bool grow)
    : CAT(lab, last->key(), grow),
      xsubject_(subject),
      xchoice_()
  {}

  inline Ptr<VectorData> getsub(Ptr<VectorData> s){
    if(!!s) return s;
    return new VectorData(1, 1.0);
  }

  CHD::ChoiceData(uint val, uint Nlevels,
		  std::vector<Ptr<VectorData> > choice,
		  Ptr<VectorData> subject)
    : CAT(val, Nlevels),
      xsubject_(getsub(subject)),
      xchoice_(choice)
  {}

  CHD::ChoiceData(uint val, Ptr<CatKey> key,
	       std::vector<Ptr<VectorData> > choice,
	       Ptr<VectorData> subject)
    : CAT(val, key),
      xsubject_(getsub(subject)),
      xchoice_(choice)
  {}

  CHD::ChoiceData(const string & lab, Ptr<CatKey> key,
	       std::vector<Ptr<VectorData> > choice,
	       Ptr<VectorData> subject, bool grow)
    : CAT(lab, key, grow),
      xsubject_(getsub(subject)),
      xchoice_(choice)
  {}

  CHD::ChoiceData(uint val, Ptr<ChoiceData> last,
	       std::vector<Ptr<VectorData> > choice,
	       Ptr<VectorData> subject)
    : CAT(val, last->key()),
      xsubject_(getsub(subject)),
      xchoice_(choice)
  {}

  CHD::ChoiceData(const string & lab, Ptr<ChoiceData> last,
		  std::vector<Ptr<VectorData> > choice,
		  Ptr<VectorData> subject, bool grow)
    : CAT(lab, last->key(), grow),
      xsubject_(getsub(subject)),
      xchoice_(choice)
  {}
  //------------------------------------------------------------
  CHD::ChoiceData(const CHD &rhs)
    : Data(rhs),
      CAT(rhs),
      xsubject_(rhs.xsubject_->clone()),
      xchoice_(rhs.xchoice_.size()),
      avail_(rhs.avail_),
      bigX(rhs.bigX)
  {
    uint n = rhs.xsubject_->size();
    for(uint i=0; i<n; ++i) xchoice_[i] = rhs.xchoice_[i]->clone();
  }

  CHD * CHD::clone()const{return new CHD(*this);}

  //======================================================================

  ostream & CHD::display(ostream &out)const{
    out << CAT::display(out) << " " << *xsubject_ << " ";
    for(uint i=0; i<xchoice_.size(); ++i) out << Xchoice(i) << " ";
    return out;
  }

//   istream & CHD::read(istream &in){
//     CAT::read(in);
//     xsubject_->read(in);
//     for(uint i=0; i<xchoice_.size(); ++i) xchoice_[i]->read(in);
//     return in;
//   }

  uint CHD::size(bool minimal)const{
    uint ans = CAT::size(minimal);
    ans += xsubject_->size(minimal);
    for(uint i=0; i<xchoice_.size(); ++i)
      ans += Xchoice(i).size();
    return ans;
  }


  uint CHD::nchoices()const{ return CAT::nlevels();}
  uint CHD::n_avail()const{return avail_.nvars();}
  bool CHD::avail(uint i)const{return avail_[i];}

  uint CHD::subject_nvars()const{return xsubject_->size();}
  uint CHD::choice_nvars()const{
    if(xchoice_.empty()) return 0;
    return xchoice_[0]->size();}

  const uint & CHD::value()const{return CAT::value();}
  void CHD::set_y(uint y){CAT::set(y);}

  const string & CHD::lab()const{return CAT::lab();}

  const std::vector<string> & CHD::labels()const{
    return CAT::labels();}

  void CHD::set_y(const string & y){CAT::set(y);}

  const Vec & CHD::Xsubject()const{return xsubject_->value();}

  const Vec & CHD::Xchoice(uint i)const{
    if(xchoice_.size()>0) return xchoice_[i]->value();
    else return null_;
  }

  void CHD::set_Xsubject(const Vec &x){
    xsubject_->set(x);
  }

  void CHD::set_Xchoice(const Vec &x, uint i){
    xchoice_[i]->set(x);
  }

  const Mat & CHD::write_x(Mat &X, bool inc_zero)const{
    bool inc = inc_zero;
    uint pch= choice_nvars();
    uint psub=  subject_nvars();
    uint M = nchoices();
    uint nc = pch +  (inc ? M : M-1)*psub;
    X.resize(M,nc);
    X=0;

    const Vec & xcu(Xsubject());
    for(uint m=0; m<M; ++m){
      const Vec & xch(Xchoice(m));
      LinAlg::VectorViewIterator it = X.row_begin(m);
      if(inc || m>0){
	it+= (inc ? m : m-1)*psub;
	std::copy(xcu.begin(), xcu.end(), it); }
      it = X.row_begin(m) + (inc ? M : M-1) *psub;
      std::copy(xch.begin(), xch.end(), it);
    }
    return X;
  }

  const Mat & CHD::X(bool inc_zeros)const{
    if(!bigX){ bigX.reset(new Mat); }
    write_x(*bigX,inc_zeros);
    return *bigX;
  }

  void CHD::set_wsp(boost::shared_ptr<Mat> newX){
    bigX = newX;
  }

  void CHD::ref_x(const ChoiceData &rhs){
    xsubject_ = rhs.xsubject_;
    xchoice_ = rhs.xchoice_;
    avail_ = rhs.avail_;
    bigX = rhs.bigX;
  }
}
