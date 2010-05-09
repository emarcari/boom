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
#include "SubMatrix.hpp"
namespace BOOM{
  namespace LinAlg{
    typedef SubMatrix SM;
    SM::SubMatrix(Matrix &m, uint rlo, uint rhi, uint clo, uint chi)
      : cols_(chi-clo + 1),
	nr_(rhi-rlo+1),
	nc_(chi-clo+1),
	stride(m.nrow())
    {
      assert(rhi >= rlo   && chi >= clo);
      assert(rhi < m.nrow() && chi < m.ncol());

      double *dat = m.data();
      uint indx=rlo+clo*stride;
      for(uint i=0; i<nc_; ++i, indx+=stride){
	cols_[i] = &dat[indx];
      }
    }

    SM::SubMatrix(const SM &rhs)
      : cols_(rhs.cols_),
	nr_(rhs.nr_),
	nc_(rhs.nc_),
	stride(rhs.stride)
    {}

    SM & SM::operator=(const SM &rhs){
      if(&rhs!=this){
	cols_ = rhs.cols_;
	nr_ = rhs.nr_;
	nc_ = rhs.nc_;
	stride = rhs.stride;
      }
      return *this;
    }


    SM & SM::operator=(const Matrix &rhs){
      assert(rhs.nrow()==nr_ && rhs.ncol()==nc_);
      for(uint i=0; i<nc_; ++i){
	std::copy(rhs.col_begin(i), rhs.col_end(i), cols_[i]);
      }
      return *this;
    }

    //------------------------------------------------------------
    uint SM::nrow()const{return nr_;}
    uint SM::ncol()const{return nc_;}
    //------------------------------------------------------------
    VectorView SM::col(uint j){
      VectorView ans(cols_[j], nr_, 1);
      return ans;
    }
    ConstVectorView SM::col(uint j)const{
      ConstVectorView ans(cols_[j], nr_, 1);
      return ans;
    }
    //------------------------------------------------------------
    VectorView SM::row(uint i){
      VectorView ans(cols_[0] + i, nc_, stride);
      return ans;
    }
    ConstVectorView SM::row(uint i)const{
      ConstVectorView ans(cols_[0]+i, nc_, stride);
      return ans;
    }
    //------------------------------------------------------------
    double SM::sum()const{
      double ans=0;
      for(uint i=0; i<nc_; ++i) ans += col(i).sum();
      return ans;
    }
    //------------------------------------------------------------

    double & SM::operator()(uint i, uint j){
      assert(i<nr_ &&  j < nc_);
      return cols_[i][j];
    }
    //------------------------------------------------------------
    const double & SM::operator()(uint i, uint j)const{
      assert(i<nr_ &&  j < nc_);
      return cols_[i][j];
    }
    //------------------------------------------------------------
    double * SM::col_begin(uint j){ return cols_[j]; }
    double * SM::col_end(uint j){ return cols_[j] + nr_; }

    const double * SM::col_begin(uint j)const{ return cols_[j]; }
    const double * SM::col_end(uint j)const{ return cols_[j] + nr_; }
    //------------------------------------------------------------

    SM & SM::operator+=(const Matrix &rhs){
      assert(rhs.nrow()==nr_ && rhs.ncol()==nc_);
      for(uint i=0; i<nc_; ++i){
	VectorView v(cols_[i], nr_, 1);
	v+= rhs.col(i);
      }
      return *this;
    }

    SM & SM::operator-=(const Matrix &rhs){
      assert(rhs.nrow()==nr_ && rhs.ncol()==nc_);
      for(uint i=0; i<nc_; ++i){
	VectorView v(cols_[i], nr_, 1);
	v-= rhs.col(i);
      }
      return *this;
    }
    //======================================================================
    typedef ConstSubMatrix CSM;
    CSM::ConstSubMatrix(const Matrix &m, uint rlo, uint rhi, uint clo, uint chi)
      : cols_(chi-clo + 1),
	nr_(rhi-rlo+1),
	nc_(chi-clo+1),
	stride(m.nrow())
    {
      assert(rhi >= rlo   && chi >= clo);
      assert(rhi < m.nrow() && chi < m.ncol());

      const double *dat = m.data();
      uint indx=rlo+clo*stride;
      for(uint i=0; i<nc_; ++i, indx+=stride){
	cols_[i] = &dat[indx];
      }
    }

    uint CSM::nrow()const{return nr_;}
    uint CSM::ncol()const{return nc_;}
    const double & CSM::operator()(uint i, uint j)const{
      assert(i<nr_ &&  j < nc_);
      return cols_[i][j];
    }
    //------------------------------------------------------------
    const double * CSM::col_begin(uint j)const{ return cols_[j]; }
    const double * CSM::col_end(uint j)const{ return cols_[j] + nr_; }
    ConstVectorView CSM::col(uint j)const{
      ConstVectorView ans(cols_[j], nr_, 1);
      return ans;
    }
    ConstVectorView CSM::row(uint i)const{
      ConstVectorView ans(cols_[0]+i, nc_, stride);
      return ans;
    }
    //------------------------------------------------------------
    double CSM::sum()const{
      double ans=0;
      for(uint i=0; i<nc_; ++i) ans += col(i).sum();
      return ans;
    }
    //------------------------------------------------------------

    
  }  // namespace LinAlg
} // namespace BOOM

