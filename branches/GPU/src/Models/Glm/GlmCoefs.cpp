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

#include "GlmCoefs.hpp"
#include <boost/bind.hpp>
#include <stdexcept>

namespace BOOM{

  GlmCoefs::GlmCoefs(uint p, bool all) // 0..p-1
    : VectorParams(p),
      inc_(p, all),
      beta_current_(false)
  {
    if(!all) add(0); // start with intercept
  }

  GlmCoefs::GlmCoefs(const Vec &b, bool infer_model_selection)
    : VectorParams(b),
      inc_(b.size()),
      beta_current_(false)
  {
    if(infer_model_selection) inc_from_beta(b);
  }

  GlmCoefs::GlmCoefs(const Vec &b, const Selector &Inc)
    : VectorParams(b),
      inc_(Inc),
      beta_current_(false)
  {
    //    assert(Inc.nvars_possible()==b.size());
    uint n = inc_.nvars();
    uint N = inc_.nvars_possible();

    if(n>N){
      ostringstream err;
      err << "Something has gone horribly wrong building "
	  << "GlmCoefs.  nvars_possible = " << N
	  << " but nvars = " << n << ".  explain that one."
	  << endl;
      throw std::runtime_error(err.str());
    }
    uint p = b.size();
    if(p>N){
      ostringstream err;
      err << "cannot build GlmCoefs with vector of size "
	  << p << " and 'Selector' of size "
	  << N << ". " << endl;
      throw std::runtime_error(err.str());
    }

    if(p<N){
      if(p==n){
	VectorParams::set(Inc.expand(b), false);
      }else{
	ostringstream err;
	err << "size of 'b' passed to constructor for GlmCoefs "
	    << " (" << p << ") must match either nvars ("
	    << n << ") or nvars_possible (" << N
	    << ")." << endl;
	throw std::runtime_error(err.str());
      }
    }
  }

  GlmCoefs::GlmCoefs(const GlmCoefs &rhs)
    : Data(rhs),
      Params(rhs),
      VectorParams(rhs),
      inc_(rhs.inc_),
      vnames_(rhs.vnames_),  // pointer semantics for vnames
      beta_current_(false)
  {
  }

  GlmCoefs * GlmCoefs::clone()const{ return new GlmCoefs(*this); }

  //-------------- model selection -------------

  const Selector & GlmCoefs::inc()const{ return inc_;}

  bool GlmCoefs::inc(uint p)const{ return inc_[p];}

  void GlmCoefs::set_inc(const Selector &new_inc){
    assert(new_inc.nvars_possible() == inc_.nvars_possible());
    uint n = nvars();
    for(uint i=0; i<n; ++i){
      uint I = indx(i);
      if(!new_inc[I]) Beta(I)=0;
    }
    inc_ = new_inc;
  }

  void GlmCoefs::add(uint i){
    beta_current_ = false;
    inc_.add(i); }

  void GlmCoefs::drop(uint i){
    inc_.drop(i);
    Beta(i)=0;
  }

  void GlmCoefs::flip(uint i){
    if(inc_[i]) drop(i);
    else add(i);
  }

  void GlmCoefs::drop_all(){
    inc_.drop_all();
    set_Beta(Vec(nvars_possible()),false);
  }

  void GlmCoefs::add_all(){
    inc_.add_all();
  }

  //------------------- size querries ----------------

  uint GlmCoefs::nvars()const{return inc().nvars();}
  uint GlmCoefs::nvars_possible()const{ return inc().nvars_possible(); }
  uint GlmCoefs::nvars_excluded()const{ return inc().nvars_excluded(); }
  uint GlmCoefs::size(bool minimal)const{
    return minimal ? nvars() : nvars_possible();}

  //-------------------- prediction ------------
  double GlmCoefs::predict(const Vec &x)const{
    uint nx = x.size();
    uint nb = nvars();
    uint Nb = nvars_possible();

//     if(nx==nb || nx == nb+1 ){
//       bool implicit_intercept(nx==nb+1);
//       const Vec &b(beta());
//       return implicit_intercept? b.affdot(x) : b.dot(x);
//     }
//     double ans=0;
//     const Vec &b(Beta());
//     if(nx==Nb || nx == Nb-1){
//       for(uint i=0; i<Nb; ++i) ans+= b[i]*x[indx(i)];
//       return ans;
//     }else if (nx==Nb-1){
//       ans = b[0];
//       for(uint i=0; i<nx; ++i) ans+= b[i+1]*x[indx(i)];
//       return ans;
//     }
//     incompatible_covariates(x, "GlmCoefs::predict");
//     return ans;

    if(nx==Nb) return x.dot(Beta());
    else if(nx==nb) return x.dot(beta());
    else incompatible_covariates(x, "GlmCoefs::predict");
    return 0;
  }

  //------ operations for included variables -----

  Vec GlmCoefs::beta()const{
    if(!beta_current_) fill_beta();
    return beta_;}

  double GlmCoefs::beta(uint i)const{
    if(!beta_current_) fill_beta();
    return beta_[i];}

  void GlmCoefs::set_beta(const Vec &b){
    if(b.size()!=nvars()) wrong_size_beta(b);
    set_Beta(inc_.expand(b), false);
//     Vec Beta(nvars_possible(),0);
//     uint n = b.size();
//     for(uint i=0; i<n; ++i){
//       uint I = indx(i);
//       Beta[I] = b[i];
//     }
//     set_Beta(Beta, false);
  }

  string GlmCoefs::vnames(uint i)const{
    if(!!vnames_){
      assert( vnames_->size() == beta().size()-1);
      uint I = indx(i);
      return (*vnames_)[I-1]; }
    if(i==0) return "Intercept";
    ostringstream out;
    out << "V" <<indx(i);
    return out.str();
  }

  std::vector<string> GlmCoefs::vnames()const{
    std::vector<string> ans;
    ans.reserve(nvars());
    for(uint i=0; i<nvars(); ++i) ans[i] = vnames(i);
    return ans;
  }

  //------- operations on all possible variables ------

  const Vec & GlmCoefs::Beta()const{
    return VectorParams::value();   }

  double GlmCoefs::Beta(uint i)const{
    return VectorParams::value()[i]; }

  double & GlmCoefs::Beta(uint i){
    beta_current_ = false;
    return VectorParams::operator[](i);}


  void GlmCoefs::set_Beta(const Vec &tmp, bool reset_inc){
    beta_current_ = false;
    VectorParams::set(tmp);
    if(reset_inc) inc_from_beta(tmp);
  }

  string GlmCoefs::Vnames(uint i)const{
    if(!!vnames_) return (*vnames_)[i];
    ostringstream out;
    out << "V" << i;
    return out.str();
  }
  std::vector<string> GlmCoefs::Vnames()const{
    if(!!vnames_) return *vnames_;
    std::vector<string> ans(nvars_possible());
    ans[0] = "Intercept";
    for(uint i=1; i<nvars_possible(); ++i){
      ostringstream out;
      out << "V" << i;
      ans[i] = out.str();}
    return ans;
  }

  void GlmCoefs::set_vnames(const std::vector<string> &vn){
    vnames_ = new std::vector<string>(vn);}

  void GlmCoefs::set_vnames(vnPtr vn){ vnames_ = vn; }

  //------- virtual function overloads ---------------

//   istream & GlmCoefs::read(istream & in){
//     VectorParams::read(in);
//     inc_from_beta(Beta());
//     beta_current_ = false;
//     return in;
//   }

  Vec GlmCoefs::vectorize(bool minimal)const{
    if(minimal) return beta();
    return VectorParams::vectorize();
  }

  Vec::const_iterator GlmCoefs::unvectorize(Vec::const_iterator &v, bool minimal){
    beta_current_ = false;
    if(minimal){
      beta_.resize(nvars());
      Vec::const_iterator e = v+beta_.size();
      std::copy(v,e, beta_.begin());
      set_beta(beta_);
      return e;
    }
    return VectorParams::unvectorize(v);
  }

  Vec::const_iterator GlmCoefs::unvectorize(const Vec &v, bool min){
    Vec::const_iterator b = v.begin();
    return unvectorize(b, min);}


  //____________________ now the private stuff ___________

  void GlmCoefs::inc_from_beta(const Vec &b){
    uint n = b.size();
    for(uint i =0; i<n; ++i){
      if(b[i]!=0) add(i);
      else drop(i);}
  }

  void GlmCoefs::wrong_size_beta(const Vec &b)const{
    ostringstream msg;
    msg << "wrong size argument given to set_beta" << endl
	<< "current size  = " << nvars() << endl
	<< "argument size = " << b.size() << endl;
    throw std::runtime_error(msg.str());
  }


  void GlmCoefs::fill_beta()const{
    beta_ = inc_.select(Beta());
    beta_current_ = true;
  }


  void GlmCoefs::incompatible_covariates(const Vec &x, const string &fname)const{
    ostringstream msg;
    msg << "incompatible covariates in " << fname << endl
	<< "beta = " << Beta() << endl
	<< "x = " << x << endl;
    throw std::runtime_error(msg.str());
  }

}
