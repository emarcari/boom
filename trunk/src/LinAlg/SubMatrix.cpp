/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "SubMatrix.hpp"
namespace BOOM{
  namespace {
    typedef SubMatrix SM;
    typedef ConstSubMatrix CSM;
  }
  SM::SubMatrix(Matrix &m, uint rlo, uint rhi, uint clo, uint chi)
      : start_(m.data() + rlo + clo*m.nrow()),
        nr_(rhi-rlo+1),
        nc_(chi-clo+1),
        stride(m.nrow())
  {
    assert(rhi >= rlo   && chi >= clo);
    assert(rhi < m.nrow() && chi < m.ncol());
  }

  SM::SubMatrix(Matrix &m)
      : start_(m.data()),
        nr_(m.nrow()),
        nc_(m.ncol()),
        stride(m.nrow())
  {}

  SM::SubMatrix(double *v, int nrow, int ncol)
      : start_(v),
        nr_(nrow),
        nc_(ncol),
        stride(nrow)
  {}

// TODO(stevescott):  need a unit test here
  SM::SubMatrix(SM &m, uint rlo, uint rhi, uint clo, uint chi)
      : start_(m.start_ + rlo + clo * m.stride),
        nr_(rhi - rlo + 1),
        nc_(chi - clo + 1),
        stride(m.stride)
  {}

  SM::SubMatrix(const SM &rhs)
      : start_(rhs.start_),
        nr_(rhs.nr_),
        nc_(rhs.nc_),
        stride(rhs.stride)
  {}

  SM & SM::operator=(const SM &rhs){
    assert(rhs.nrow()==nr_ && rhs.ncol()==nc_);
    for(uint i=0; i<nc_; ++i){
      std::copy(rhs.col_begin(i), rhs.col_end(i), cols(i));
    }
    return *this;
  }

  SM & SM::reset(const SM &rhs){
    if(&rhs!=this){
      start_ = rhs.start_;
      nr_ = rhs.nr_;
      nc_ = rhs.nc_;
      stride = rhs.stride;
    }
    return *this;
  }


  SM & SM::operator=(const Matrix &rhs){
    assert(rhs.nrow()==nr_ && rhs.ncol()==nc_);
    for(uint i=0; i<nc_; ++i){
      std::copy(rhs.col_begin(i), rhs.col_end(i), cols(i));
    }
    return *this;
  }

  //------------------------------------------------------------
  uint SM::nrow()const{return nr_;}
  uint SM::ncol()const{return nc_;}
  //------------------------------------------------------------
  VectorView SM::col(uint j){
    VectorView ans(cols(j), nr_, 1);
    return ans;
  }
  ConstVectorView SM::col(uint j)const{
    ConstVectorView ans(cols(j), nr_, 1);
    return ans;
  }
  //------------------------------------------------------------
  VectorView SM::row(uint i){
    VectorView ans(cols(0) + i, nc_, stride);
    return ans;
  }
  ConstVectorView SM::row(uint i)const{
    ConstVectorView ans(cols(0)+i, nc_, stride);
    return ans;
  }

  VectorView SM::diag(){
    int m = std::min(nr_, nc_);
    return VectorView(cols(0), m, stride+1);
  }
  ConstVectorView SM::diag()const{
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0), m, stride+1);
  }

  VectorView SM::subdiag(int i){
    if(i < 0) return superdiag(-i);
    int m = std::min(nr_, nc_);
    return VectorView(cols(0) + i, m-i, stride+1);
  }

  ConstVectorView SM::subdiag(int i)const{
    if(i < 0) return superdiag(-i);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0) + i, m-i, stride+1);
  }

  VectorView SM::superdiag(int i){
    if(i < 0) return subdiag(-1);
    int m = std::min(nr_, nc_);
    return VectorView(cols(i), m-i, stride+1);
  }

  ConstVectorView SM::superdiag(int i)const{
    if(i < 0) return subdiag(-1);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(i), m-i, stride+1);
  }

  //------------------------------------------------------------
  double SM::sum()const{
    double ans=0;
    for(uint i=0; i<nc_; ++i) ans += col(i).sum();
    return ans;
  }
  //------------------------------------------------------------
  Matrix SM::to_matrix()const{
    Matrix ans(nrow(), ncol());
    for(int i = 0; i < ncol(); ++i){
      ans.col(i) = this->col(i);
    }
    return ans;
  }
  //------------------------------------------------------------

  double & SM::operator()(uint i, uint j){
    assert(i<nr_ &&  j < nc_);
    //      return cols(i)[j];
    return cols(j)[i];
  }
  //------------------------------------------------------------
  const double & SM::operator()(uint i, uint j)const{
    assert(i<nr_ &&  j < nc_);
    //      return cols(i)[j];
    return cols(j)[i];
  }
  //------------------------------------------------------------
  double * SM::col_begin(uint j){ return cols(j); }
  double * SM::col_end(uint j){ return cols(j) + nr_; }

  const double * SM::col_begin(uint j)const{ return cols(j); }
  const double * SM::col_end(uint j)const{ return cols(j) + nr_; }
  //------------------------------------------------------------

  SM & SM::operator+=(const Matrix &rhs){
    assert(rhs.nrow()==nr_ && rhs.ncol()==nc_);
    for(uint i=0; i<nc_; ++i){
      VectorView v(cols(i), nr_, 1);
      v+= rhs.col(i);
    }
    return *this;
  }

  SM & SM::operator-=(const Matrix &rhs){
    assert(rhs.nrow()==nr_ && rhs.ncol()==nc_);
    for(uint i=0; i<nc_; ++i){
      VectorView v(cols(i), nr_, 1);
      v-= rhs.col(i);
    }
    return *this;
  }

  SM & SM::operator+=(double x){
    for(uint i = 0; i < nc_; ++i){
      col(i) += x;
    }
    return *this;
  }

  SM & SM::operator-=(double x){
    for(uint i = 0; i < nc_; ++i){
      col(i) -= x;
    }
    return *this;
  }

  SM & SM::operator*=(double x){
    for(uint i = 0; i < nc_; ++i){
      col(i) *= x;
    }
    return *this;
  }

  SM & SM::operator/=(double x){
    for(uint i = 0; i < nc_; ++i){
      col(i) /= x;
    }
    return *this;
  }

  //======================================================================
  CSM::ConstSubMatrix(const Matrix &m, uint rlo, uint rhi, uint clo, uint chi)
      : start_(m.data() + clo*m.nrow() + rlo),
        nr_(rhi-rlo+1),
        nc_(chi-clo+1),
        stride(m.nrow())
  {
    assert(rhi >= rlo   && chi >= clo);
    assert(rhi < m.nrow() && chi < m.ncol());
  }

  uint CSM::nrow()const{return nr_;}
  uint CSM::ncol()const{return nc_;}
  const double & CSM::operator()(uint i, uint j)const{
    assert(i<nr_ &&  j < nc_);
    return cols(j)[i];
  }
  //------------------------------------------------------------
  const double * CSM::col_begin(uint j)const{ return cols(j); }
  const double * CSM::col_end(uint j)const{ return cols(j) + nr_; }
  ConstVectorView CSM::col(uint j)const{
    ConstVectorView ans(cols(j), nr_, 1);
    return ans;
  }
  ConstVectorView CSM::row(uint i)const{
    ConstVectorView ans(cols(0)+i, nc_, stride);
    return ans;
  }

  ConstVectorView CSM::diag()const{
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0), m, stride+1);
  }

  ConstVectorView CSM::subdiag(int i)const{
    if(i < 0) return superdiag(-i);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0) + i, m-i, stride+1);
  }

  ConstVectorView CSM::superdiag(int i)const{
    if(i < 0) return subdiag(-1);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(i), m-i, stride+1);
  }


  //------------------------------------------------------------
  double CSM::sum()const{
    double ans=0;
    for(uint i=0; i<nc_; ++i) ans += col(i).sum();
    return ans;
  }
  //------------------------------------------------------------

  namespace {
    template <class M1, class M2>
    bool MatrixEquals(const M1 &lhs, const M2 &rhs){
      if(lhs.nrow() != rhs.nrow()) return false;
      if(lhs.ncol() != rhs.ncol()) return false;
      for(int i = 0; i < lhs.nrow(); ++i){
        for(int j = 0; j < lhs.ncol(); ++j){
          if(lhs(i, j) != rhs(i, j)) return false;
        }
      }
      return true;
    }
  }

  bool operator==(const Matrix &lhs, const SubMatrix &rhs){
    return MatrixEquals(lhs, rhs);
  }
    bool operator==(const Matrix &lhs, const ConstSubMatrix &rhs){
    return MatrixEquals(lhs, rhs);
  }
    bool operator==(const SubMatrix &lhs, const Matrix &rhs){
    return MatrixEquals(lhs, rhs);
  }
    bool operator==(const SubMatrix &lhs, const SubMatrix &rhs){
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const SubMatrix &lhs, const ConstSubMatrix &rhs){
    return MatrixEquals(lhs, rhs);
  }
    bool operator==(const ConstSubMatrix &lhs, const Matrix &rhs){
    return MatrixEquals(lhs, rhs);
  }
    bool operator==(const ConstSubMatrix &lhs, const SubMatrix &rhs){
    return MatrixEquals(lhs, rhs);
  }
    bool operator==(const ConstSubMatrix &lhs, const ConstSubMatrix &rhs){
    return MatrixEquals(lhs, rhs);
  }

} // namespace BOOM
