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
// std library includes
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <numeric>

#include "VectorView.hpp"
#include "Vector.hpp"
#include <distributions.hpp>
extern "C"{
#include <cblas.h>
}

namespace BOOM{
  using namespace std;

  namespace{
    template <class V1, class V2> double dot_impl(
        const V1 &v1, const V2 &v2) {
      assert(v1.size() == v2.size());
      if(v1.stride() > 0 && v2.stride() > 0){
        return cblas_ddot(v1.size(),
                          v1.data(), v1.stride(),
                          v2.data(), v2.stride());
      }else{
        double ans = 0;
        for(int i = 0; i < v1.size(); ++i){
          ans += v1[i] * v2[i];
        }
        return ans;
      }
    }
  }


  typedef VectorView VV;

  VV::iterator VV::begin(){return iterator(V, V, stride()); }
  VV::iterator VV::end(){
    return iterator(V+size()*stride(), V, stride()); }
  VV::const_iterator VV::begin()const{
    return const_iterator(V, V, stride()); }
  VV::const_iterator VV::end()const{
    return const_iterator(V+size()*stride(), V, stride()); }

  VV::reverse_iterator VV::rbegin(){
    return std::reverse_iterator<iterator>(begin());}
  VV::reverse_iterator VV::rend(){
    return std::reverse_iterator<iterator>(end());}
  VV::const_reverse_iterator VV::rbegin()const{
    return std::reverse_iterator<const_iterator>(begin());}
  VV::const_reverse_iterator VV::rend()const{
    return std::reverse_iterator<const_iterator>(end());}


  VV::VectorView(double *first, uint n, int s)
      : V(first),
      nelem_(n),
      stride_(s)
      {}

  VV & VV::reset(double *first, uint n, uint s){
    V = first;
    nelem_ = n;
    stride_ = s;
    return *this;
  }

  VV::VectorView(Vector &v, uint first)
      : V(v.data()+first),
      nelem_(v.size()-first),
      stride_(1)
      {}

  VV::VectorView(Vector &v, uint first, uint len)
      : V(v.data()+first),
      nelem_(len),
      stride_(1)
      {}

  VV::VectorView(VectorView v, uint first, uint len)
      : V(v.data() + first * v.stride()),
      nelem_(len),
      stride_(v.stride())
      {}

  VV & VV::operator=(double x){
    std::fill(begin(), end(), x);
    return *this;
  }

  VV & VV::operator=(const Vector &x){
    assert(x.size()==size());
    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  VV & VV::operator=(const VectorView &x){
    assert(x.size()==size());
    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  VV & VV::operator=(const ConstVectorView &x){
    assert(x.size()==size());
    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  void VV::randomize(){
    uint n = size();
    double *d = data();
    for(uint i=0; i<n; ++i) d[i] = runif(0,1);
  }

  VV & VV::operator+=(const double & x){
    VV &A(*this);
    for(uint i=0; i<size(); ++i) A[i]+=x;
    return *this; }

  VV & VV::operator-=(const double & x){
    VV &A(*this);
    for(uint i=0; i<size(); ++i) A[i]-=x;
    return *this; }

  VV & VV::operator*=(const double & x){
    cblas_dscal(size(), x, data(), stride());
    return *this; }

  VV & VV::operator/=(const double & x){
    assert(x!=0.0);
    cblas_dscal(size(), 1.0/x, data(), stride());
    return *this; }

  VV & VV::operator+=(const VectorView & y){
    assert(y.size()==size());
    cblas_daxpy(size(), 1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV & VV::operator+=(const ConstVectorView & y){
    assert(y.size()==size());
    cblas_daxpy(size(), 1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV & VV::operator+=(const Vector &y){
    assert(y.size()==size());
    cblas_daxpy(size(), 1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV & VV::operator-=(const Vector &y){
    assert(y.size()==size());
    cblas_daxpy(size(), -1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV & VV::operator-=(const VectorView &y){
    assert(y.size()==size());
    cblas_daxpy(size(), -1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV & VV::operator-=(const ConstVectorView &y){
    assert(y.size()==size());
    cblas_daxpy(size(), -1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV & VV::axpy(const Vector &y, double a){
    assert(y.size()==size());
    cblas_daxpy(size(), a, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV & VV::axpy(const VectorView &y, double a){
    assert(y.size()==size());
    cblas_daxpy(size(), a, y.data(), y.stride(), data(), stride());
    return *this;
  }

  inline void dmul(uint n, double *x, uint xs, const double *y, uint ys ){
    for(uint i=0 ; i<n; ++i){
      *x *= *y;
      x+= xs;
      y+= ys;}}


  VV & VV::operator*=(const Vector &y){
    assert(size()==y.size());
    dmul(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV & VV::operator*=(const VectorView &y){
    assert(size()==y.size());
    dmul(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV & VV::operator*=(const ConstVectorView &y){
    assert(size()==y.size());
    dmul(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  inline void ddiv(uint n, double *x, uint xs, const double *y, uint ys ){
    for(uint i=0; i<n; ++i){
      *x /= *y;
      x+= xs;
      y+= ys;}}


  VV & VV::operator/=(const Vector &y){
    assert(size()==y.size());
    ddiv(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV & VV::operator/=(const VectorView &y){
    assert(size()==y.size());
    ddiv(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV & VV::operator/=(const ConstVectorView &y){
    assert(size()==y.size());
    ddiv(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  double VV::normsq()const{
    double tmp = cblas_dnrm2(size(), data(), stride());
    return tmp*tmp;
  }

  double VV::normalize_prob(){
    double s = cblas_dasum(size(), data(), stride());
    if(s==0) throw_exception<runtime_error>("normalizing constant is zero in VV::normalize_logprob");
    operator/=(s);
    return s;
  }

  double VV::normalize_logprob(){
    double nc=0;
    VectorView &x= *this;
    double m = max();
    uint n = size();
    for(uint i=0; i<n; ++i){
      x[i] = std::exp(x[i]-m);
      nc+=x[i]; }
    x/=nc;
    return nc;   // might want to change this
  }


  double VV::min()const{
    const_iterator it = min_element(begin(), end());
    return *it; }

  double VV::max()const{
    const_iterator it = max_element(begin(), end());
    return *it; }

  uint VV::imax()const{
    const_iterator it = max_element(begin(), end());
    return it-begin();}

  uint VV::imin()const{
    const_iterator it = min_element(begin(), end());
    return it-begin();}

  double VV::sum()const{
    return accumulate(begin(), end(), 0.0); }

  double VV::abs_norm()const{
    return cblas_dasum(size(), data(), stride());}

  inline double mul(double x, double y){return x*y;}
  double VV::prod()const{
    return accumulate(begin(), end(), 1.0, mul);}

  double VV::dot(const Vector &y)const{return dot_impl(*this, y); }
  double VV::dot(const VectorView &y)const{return dot_impl(*this, y); }
  double VV::dot(const ConstVectorView &y)const{return dot_impl(*this, y); }

  double VV::affdot(const Vector &y)const{
    uint n = size();
    uint m = y.size();
    if(m==n) return dot(y);
    double ans=0.0;
    const double *v1=0, *v2=0;
    if(m==n+1){    // y is one unit longer than x
      ans= y.front();
      v1 = y.data()+1;
      v2 = data();
    }else if (n==m+1){   // x is one unit longer than y
      ans = front();
      v1 = y.data();
      v2 = data()+1;
    }else{
      throw_exception<runtime_error>("x and y do not conform in affdot");
    }
    const int i(std::min(m,n));
    return cblas_ddot(i, v1, y.stride(), v2, stride());
  }


  double VV::affdot(const VectorView &y)const{
    uint n = size();
    uint m = y.size();
    if(m==n) return dot(y);
    double ans=0.0;
    const double *v1=0, *v2=0;
    if(m==n+1){    // y is one unit longer than x
      ans= y.front();
      v1 = y.data()+1;
      v2 = data();
    }else if (n==m+1){   // x is one unit longer than y
      ans = front();
      v1 = y.data();
      v2 = data()+1;
    }else{
      throw_exception<runtime_error>("x and y do not conform in affdot");
    }
    const int i(std::min(m,n));
    return cblas_ddot(i, v1, y.stride(), v2, stride());
  }


  ostream & operator<<(ostream & out, const VV & v){
    for(uint i = 0; i<v.size(); ++i) out << v[i] << " ";
    return out; }

  istream & operator<< (istream &in, VV &v){
    for(uint i=0; i<v.size(); ++i) in >> v[i];
    return in;
  }

  //======================================================================

  typedef ConstVectorView CVV;

  CVV::const_iterator CVV::begin()const{return const_iterator(V, V, stride()); }
  CVV::const_iterator CVV::end()const{return const_iterator(V+size()*stride(), V, stride()); }

  CVV::const_reverse_iterator CVV::rbegin()const{
    return std::reverse_iterator<const_iterator>(begin());}
  CVV::const_reverse_iterator CVV::rend()const{
    return std::reverse_iterator<const_iterator>(end());}


  CVV::ConstVectorView(const double *first, uint n, int s)
      : V(first),
      nelem_(n),
      stride_(s)
      {}

  CVV::ConstVectorView(const Vector &v, uint first)
      : V(v.data() + first),
      nelem_(v.size() - first),
      stride_(1)
      {}

  CVV::ConstVectorView(const Vector &v, uint first, uint len)
      : V(v.data() + first),
      nelem_(len),
      stride_(1)
      {}

  CVV::ConstVectorView(const CVV &v, uint first)
      : V(v.data() + first * v.stride()),
      nelem_(v.size() - first),
      stride_(v.stride())
      {}

  CVV::ConstVectorView(const VectorView &v, uint first, uint len)
      : V(v.data() + first*v.stride()),
      nelem_(len),
      stride_(v.stride())
      {}

  CVV::ConstVectorView(const CVV &v, uint first, uint len)
      : V(v.data() + first*v.stride()),
      nelem_(len),
      stride_(v.stride())
      {}

  CVV::ConstVectorView(const VectorView &v)
      : V(v.data()),
      nelem_(v.size()),
      stride_(v.stride())
      {}

  double CVV::normsq()const{
    double tmp = cblas_dnrm2(size(), data(), stride());
    return tmp*tmp;
  }

  double CVV::min()const{
    const_iterator it = min_element(begin(), end());
    return *it; }

  double CVV::max()const{
    const_iterator it = max_element(begin(), end());
    return *it; }

  uint CVV::imax()const{
    const_iterator it = max_element(begin(), end());
    return it-begin();}

  uint CVV::imin()const{
    const_iterator it = min_element(begin(), end());
    return it-begin();}

  double CVV::sum()const{
    return accumulate(begin(), end(), 0.0); }

  double CVV::abs_norm()const{
    return cblas_dasum(size(), data(), stride());}

  double CVV::prod()const{
    return accumulate(begin(), end(), 1.0, mul);}

  double CVV::dot(const Vector &y)const{return dot_impl(*this, y); }
  double CVV::dot(const VectorView &y)const{return dot_impl(*this, y); }
  double CVV::dot(const ConstVectorView &y)const{return dot_impl(*this, y); }

  CVV CVV::reverse()const{
    const double *start = V + (nelem_ - 1) * stride_;
    return CVV(start, nelem_, -stride_);
  }

  ostream & operator<<(ostream & out, const CVV & v){
    for(uint i = 0; i<v.size(); ++i) out << v[i] << " ";
    return out; }

}
