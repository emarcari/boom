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

#include "include.hpp"
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>
#include <cpputil/seq.hpp>

#include <distributions.hpp>

#include <stdexcept>
#include <sstream>



namespace BOOM{
  typedef std::vector<bool> vb;
  typedef std::vector<uint> vpos;

  vb s2vb(const std::string &);

  vb s2vb(const std::string & s){
     uint n = s.size();
     std::vector<bool> ans(n,false);
     for(uint i=0; i<n; ++i){
       char c = s[i];
       if(c=='1') ans[i] = true;
       else if(c=='0') ans[i] = false;
       else{
 	ostringstream err;
 	err << "only 0's and 1's are allowed in the 'include' string constructor "
 	    << endl
 	    << "you supplied:  "  << endl
 	    << s << endl
 	    << "first illegal value found at position " << i << "." << endl;
 	throw std::runtime_error(err.str());
       }
     }
     return ans;
  }



  void include::reset_inc_indx(){
    inc_indx.clear();
    for(uint i=0; i<nvars_possible(); ++i)
      if(inc(i)) inc_indx.push_back(i);
  }

  typedef BinomialProcessData BPD;

  include::include(){}

  include::include(uint p, bool all)
    : BPD(p,all),
      include_all(all)
  {
    reset_inc_indx();
  }

  include::include(const std::string &s)
    : BPD(s2vb(s)),
      include_all(false)
  {
    if(nvars()==nvars_possible()) include_all=true;
    reset_inc_indx();
  }

  include::include(const vb& in)
    : BPD(in),
      include_all(false)
  {
    reset_inc_indx();
  }

  include::include(const std::vector<uint> &pos, uint n)
    : BPD(n, false),
      inc_indx(),
      include_all(false)
  {
    for(uint i=0; i<pos.size(); ++i) add(pos[i]);
  }

  include::include(const include &rhs)
    : Data(rhs),
      BPD(rhs),
      inc_indx(rhs.inc_indx),
      include_all(rhs.include_all)
  {}

  include & include::operator=(const include &rhs){
    if(&rhs==this) return *this;
    BPD::operator=(rhs);
    inc_indx = rhs.inc_indx;
    include_all =rhs.include_all;
    return *this;
  }

  void include::check_size_eq(uint p, const string &fun)const{
    if(p==nvars_possible()) return;
    ostringstream err;

    err << "error in function include::" << fun << endl
	<< "include::nvars_possible()== " << nvars_possible() << endl
	<< "you've assumed it to be " << p << endl
	<< "shame on you!" << endl;
    throw std::runtime_error(err.str());
  }

  void include::check_size_gt(uint p, const string &fun)const{
    if(p< nvars_possible()) return;
    ostringstream err;

    err << "error in function include::" << fun << endl
	<< "include::nvars_possible()== " << nvars_possible() << endl
	<< "you tried to access element " << p << endl
	<< "shame on you!" << endl;
    throw std::runtime_error(err.str());
  }

  include & include::add(uint p){
    check_size_gt(p, "add");
    if(include_all) return *this;
    if(inc(p)==false){
      set_bit(p, true);
      vpos::iterator it =
 	std::lower_bound(inc_indx.begin(), inc_indx.end(), p);
      inc_indx.insert(it, p);}
    return *this;
  }

  void include::drop_all(){
    include_all = false;
    inc_indx.clear();
    std::vector<bool> all(nvars_possible(), false);
    BPD::set(all);
  }

  void include::add_all(){
    include_all=true;
    uint n = nvars_possible();
    inc_indx = seq<uint>(0, n-1);
    std::vector<bool> all(n, true);
    BPD::set(all);
  }

  include &include::drop(uint p){
    check_size_gt(p, "drop");
    if(include_all){
      reset_inc_indx();
      include_all=false;
    }
    if(inc()[p]==true){
      set_bit(p, false);
      vpos::iterator it =
 	std::lower_bound(inc_indx.begin(), inc_indx.end(), p);
      inc_indx.erase(it);}
    return *this;
  }

  include & include::flip(uint p){
    if(inc(p)) drop(p);
    else add(p);
    return *this;
  }


  include include::complement()const{
    include ans(*this);
    for(uint i=0; i<nvars_possible(); ++i){
      ans.flip(i);
    }
    return ans;
  }

  void include::swap(include &rhs){
    BPD::swap(rhs);
    std::swap(inc_indx, rhs.inc_indx);
    std::swap(include_all, rhs.include_all);
  }

  void include::swap(BPD & rhs){
    BPD::swap(rhs);
    reset_inc_indx();
  }

  bool include::inc(uint i)const{return BPD::operator[](i);}
  const vb & include::inc()const{ return BPD::value();}

  uint include::nvars()const{
    return include_all ? nvars_possible() : inc_indx.size(); }

  uint include::nvars_possible()const{return BPD::size();}

  uint include::nvars_excluded()const{return nvars_possible() - nvars();}

  uint include::indx(uint i)const{
    if(include_all) return i;
    return inc_indx[i]; }
  uint include::INDX(uint i)const{
    if(include_all) return i;
    std::vector<uint>::const_iterator loc =
      std::lower_bound(inc_indx.begin(), inc_indx.end(), i);
    return loc - inc_indx.begin();
  }

  Vec include::vec()const{
    Vec ans(nvars_possible(), 0.0);
    uint n = nvars();
    for(uint i=0; i<n; ++i){
      uint I = indx(i);
      ans[I] = 1;
    }
    return ans;
  }

  bool include::covers(const include &rhs)const{
    for(uint i=0; i<rhs.nvars(); ++i){
      uint I = rhs.indx(i);
      if(!inc(I)) return false;}
    return true;}

  bool include::operator==(const include &rhs)const{
    return BPD::operator==(rhs); }
  bool include::operator!=(const include &rhs)const{
    return ! operator==(rhs);}

  include include::Union(const include &rhs)const{
    uint n = nvars_possible();
    check_size_eq(rhs.nvars_possible(), "Union");
    include ans(n, false);
    for(uint i=0; i<n; ++i) if(inc(i) || rhs.inc(i)) ans.add(i);
    return ans;
  }

  include include::intersection(const include &rhs)const{
    uint n = nvars_possible();
    check_size_eq(rhs.nvars_possible(), "intersection");
    include ans(n, false);
    const include &shorter(rhs.nvars() < nvars()? rhs: *this);
    const include &longer(rhs.nvars()<nvars() ? *this : rhs);

    for(uint i=0; i<shorter.nvars(); ++i){
      uint I = shorter.indx(i);     // I is included in shorter
      if(longer.inc(I))             // I is included in longer
	ans.add(I);
    }
    return ans;}


  include & include::cover(const include &rhs){
    check_size_eq(rhs.nvars_possible(), "cover");
    for(uint i=0; i<rhs.nvars(); ++i)
      add(rhs.indx(i));  // does nothing if already added
    return *this;
  }

  template <class V>
  Vec inc_select(const V &x, const include &inc){
    uint nx = x.size();
    uint N = inc.nvars_possible();
    if(nx != N){
      ostringstream msg;
      msg << "include::select... x.size() = " << nx << " nvars_possible() = "
	  << N << endl;
      throw std::runtime_error(msg.str());
    }
    uint n = inc.nvars();

    if(n==N) return x;
    Vec ans(n);
    for(uint i=0; i<n; ++i) ans[i] = x[inc.indx(i)];
    return ans;
  }

  Vec include::select(const Vec &x)const{
    return inc_select<Vec>(x, *this); }
  Vec include::select(const VectorView &x)const{
    return inc_select<VectorView>(x, *this); }
  Vec include::select(const ConstVectorView &x)const{
    return inc_select<ConstVectorView>(x, *this); }

  template <class V>
  Vec inc_expand(const V &x, const include &inc){
    uint n = inc.nvars();
    uint nx = x.size();
    if(nx!=n){
      ostringstream msg;
      msg << "include::expand... x.size() = " << nx << " nvars() = "
	  << n << endl;
      throw std::runtime_error(msg.str());
    }
    uint N = inc.nvars_possible();
    if(n==N) return x;
    Vec ans(N, 0);
    for(uint i=0; i<n; ++i){
      uint I = inc.indx(i);
      ans[I] = x[i];
    }
    return ans;
  }

  Vec include::expand(const Vec & x)const{
    return inc_expand(x,*this); }
  Vec include::expand(const VectorView & x)const{
    return inc_expand(x,*this); }
  Vec include::expand(const ConstVectorView & x)const{
    return inc_expand(x,*this); }


  Vec include::select_add_int(const Vec &x)const{
    assert(x.size()==nvars_possible()-1);
    //    bool need_lb(false);
    //    int lb(0);
//     if(x.lo()!=1){
//       need_lb=true;
//       lb = x.lo();
//       const_cast<Vec &>(x).set_lower_bound(1);
//     }
    if(include_all) return concat(1.0,x);
    Vec ans(nvars());
    ans[0]= inc(0) ? 1.0 : x[indx(0)-1];
    for(uint i=1; i<nvars(); ++i) ans[i] = x[indx(i)-1];
    //    for(uint i=1; i<nvars(); ++i) ans[i] = x[indx(i)];
    //    if(need_lb) const_cast<Vec &>(x).set_lower_bound(lb);
    return ans;
  }
  //----------------------------------------------------------------------
//   Spd include::select(const Spd &S)const{
//     assert(S.ncol()==nvars_possible() ||
//  	   !"S is the wrong size in include::select ");
//     if(include_all) return S;
//     Spd ans(nvars());
//     for(uint i=0; i<nvars(); ++i){
//       uint I = indx(i);
//       for(uint j=0; j<=i; ++j){
//   	uint J = indx(j);
//   	double tmp = S(I,J);
//   	ans(i,j) = tmp;
//   	if(i!=j) ans(j,i)=tmp;}}
//     return ans;
//   }
  //----------------------------------------------------------------------
  Spd include::select(const Spd &S)const{
    uint n = nvars();
    uint N = nvars_possible();
    check_size_eq(S.ncol(), "select");
    if(include_all || n==N) return S;
    Spd ans(n);
    for(uint i=0; i<n; ++i){
      uint I = inc_indx[i];
      const double * s(S.col(I).data());
      double * a(ans.col(i).data());
      for(uint j=0; j<n; ++j) a[j] = s[inc_indx[j]];
    }
    return ans;
  }
  //----------------------------------------------------------------------
  Mat include::select_cols(const Mat &m)const{
    if(include_all) return m;
    Mat ans(m.nrow(), nvars());
    for(uint i=0; i<nvars(); ++i){
      uint I=indx(i);
      std::copy(m.col_begin(I), m.col_end(I), ans.col_begin(i));
    }
    return ans;
  }

  Mat include::select_rows(const Mat &m)const{
    if(include_all) return m;
    uint n = nvars();
    Mat ans(n, m.ncol());
    for(uint i=0; i<n; ++i) ans.row(i) = m.row(indx(i));
    return ans;
  }

  Mat include::select_square(const Mat &m)const{
    assert(m.is_square());
    check_size_eq(m.nrow(), "select_square");
    if(include_all) return m;

    Mat ans(nvars(), nvars());
    for(uint i=0; i<nvars(); ++i){
      uint I = indx(i);
      for(uint j=0; j<nvars(); ++j){
 	uint J = indx(j);
 	ans(i,j) = m(I,J); }}
    return ans;
  }

  Vec & include::zero_missing_elements(Vec &v)const{
    uint N = nvars_possible();
    check_size_eq(v.size(), "zero_missing_elements");
    const include & inc(*this);
    for(uint i=0; i<N; ++i){
      if(!inc[i]) v[i] = 0;
    }
    return v;
  }

  uint include::random_included_position()const{
    assert(inc(0)); // intercept is included
    uint n = nvars();
    if(n==1) return nvars_possible()+1;
    int j = rmulti(1, n-1);
    return indx(j);
  }

  uint include::random_included_position(const Vec &probs)const{
    assert(inc(0)); // intercept is included
    uint n = nvars();
    if(n==1) return nvars_possible()+1;
    Vec p(1, static_cast<int>(nvars())-1);
    for(uint i=1; i<nvars(); ++i) p[i] = probs[indx(i)];
    int j = rmulti(p);
    return indx(j);
  }

  uint include::random_excluded_position()const{
    uint N = nvars_possible();
    uint n = nvars();
    uint nex = N-n;  // number of excluded variables
    if(nex==0) return N+1;  // no variables are excluded
    if(  (static_cast<double>(nex)/N) >=.1){
      while(true){
 	int j = rmulti(1, N-1);
 	if(!inc(j)) return j;
      }
    }else{
      uint pos = rmulti(1, nex);
      uint cnt=0;
      for(uint i=0; i<nvars_possible(); ++i){
 	if(!inc(i)){
 	  ++cnt;
 	  if(cnt==pos) return i;}}}
    return N+2; // error
  }

  uint include::random_excluded_position(const Vec &probs)const{
    uint N = nvars_possible();
    uint n = nvars();
    uint nex = N-n;  // number of excluded variables
    if(nex==0) return N+1;  // no variables are excluded
    Vec p(probs);
    for(uint i=0; i<p.size(); ++i) if(!inc(i)) p[i]=0.0;
    return rmulti(p);
  }

  include &include::operator+=(const include &rhs){
    return cover(rhs);}
  include &include::operator-=(const include &rhs){
    check_size_eq(rhs.nvars_possible(), "operator-=");
    for(uint i=0; i<rhs.nvars(); ++i) drop(rhs.indx(i));
    return *this;}

  include & include::operator*=(const include &rhs){
    include tmp = intersection(rhs);
    this->swap(tmp);
    return *this;
  }


  ostream & operator<<(ostream &out, const include &inc){
    for(uint i=0; i<inc.nvars_possible(); ++i) out << inc.inc(i);
    return out;
  }

  istream & operator>>(istream &in, include &inc){
    string s;
    in >> s;
    uint n = s.size();
    vb tmp(n);
    for(uint i=0; i<n; ++i){
      if(s[i]=='0') tmp[i]=false;
      else if(s[i]=='1') tmp[i]=true;
      else throw std::runtime_error(s+"is an illegal input value for 'include'");
    }
    inc.set(tmp);
    return in;
  }

  include operator-(const include &lhs, const include &rhs){
    assert(lhs.nvars_possible()==rhs.nvars_possible());
    include ans(lhs);
    for(uint i=0; i<rhs.nvars(); ++i) ans.drop(rhs.indx(i));
    return ans;
  }

  include operator+(const include &lhs, const include &rhs){
    return lhs.Union(rhs); }

  include operator*(const include &lhs, const include &rhs){
    return lhs.intersection(rhs);}

  //============================================================

  Vec operator*(double x, const include &inc){
    Vec ans(inc.nvars_possible(), 0.0);
    uint n = inc.nvars();
    for(uint i=0; i<n; ++i){
      uint I = inc.indx(i);
      ans[I] = x;
    }
    return ans;
  }

  Vec operator*(const include &inc, double x){ return x*inc; }


  //============================================================


  inline bool check_vec(const Vec &big, int pos, const Vec &small){
    for(uint i=0; i<small.size(); ++i){
      uint I = i;
      if(I >= big.size()) return false;
      if(big[pos+I]!=small[i]) return false;
    }
    return true;
  }

  include find_contiguous_subset(const Vec &big, const Vec &small){
    vb vec(big.size(), false);
    Vec::const_iterator b = big.begin();
    Vec::const_iterator it = big.begin();
    Vec::const_iterator end = big.end();

    for(uint i=0; i<small.size(); ++i){
      it=std::find(it,end, small[i]);
      uint I = it-b;
      vec[I]=true;
    }
    return include(vec);}


  //============================================================
  include append(bool newinc, const include &Inc){
    typedef std::vector<bool> vb;
    vb res(Inc.nvars_possible()+1);
    const vb& old(Inc.inc());
    vb::iterator it = res.begin();
    *it = newinc;
    ++it;
    std::copy(old.begin(), old.end(), it);
    return include(res);
  }

  include append(const include &Inc, bool newinc){
    typedef std::vector<bool> vb;
    vb res(Inc.nvars_possible()+1);
    const vb& old(Inc.inc());
    vb::iterator it = res.begin();
    std::copy(old.begin(), old.end(), res.begin());
    res.back()=newinc;
    return include(res);
  }

  include append(const include &Inc1, const include &Inc2){
    typedef std::vector<bool> vb;
    const vb & first(Inc1.inc());
    const vb & second(Inc2.inc());
    vb res(first.size()+second.size());
    vb::iterator resit = copy(first.begin(), first.end(), res.begin());
    copy(second.begin(), second.end(), resit);
    return include(res);
  }

}
