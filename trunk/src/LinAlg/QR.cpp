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
#include "QR.hpp"

extern "C"{
#include <cblas.h>

  void dgeqrf_(int *, int *, double *, int *, double *,
	       double *, int *, int *);
  void dorgqr_(int *, int *, int *, double *, int *, const double *,
	       const double *, const int *, int *);
  void dormqr_(const char *, const char *, int *, int *, int *, const double *,
	       int *, const double *, double *, int *, double *, int *, int *);

  void dtrtrs_(const char *,const char *,const char *, int *, int *,
	       const double *, int *, double *, int *, int *);

}

namespace BOOM{
  namespace LinAlg{
    QR::QR(const Matrix &mat)
      : dcmp(mat),
	tau(std::min(mat.nrow(), mat.ncol())),
	work(mat.ncol())
    {
      int m = mat.nrow();
      int n = mat.ncol();
      int info=0;
      lwork = -1;
      // have LAPACK compute optimal lwork...
      dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
      lwork = static_cast<int>(work[0]);
      work.resize(lwork);
      // compute the decomposition with the optimal value...
      dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
    }

    Matrix QR::getQ()const{
      Matrix ans(dcmp);
      int m = ans.nrow();
      int n = ans.ncol();
      int k = std::min(m,n);
      int info=0;
      dorgqr_(&m, &n, &k, ans.data(), &m, tau.data(), work.data(),
	      &lwork, &info);
      return ans;
    }

    Matrix QR::getR()const{
      uint m = dcmp.nrow();
      uint n = dcmp.ncol();
      uint k = std::min(m,n);
      Matrix ans(k,n, 0.0);
      if(m>=n){  // usual case
	for(uint i=0; i<n; ++i)
	  std::copy(dcmp.col_begin(i), dcmp.col_begin(i)+i+1,
		    ans.col_begin(i));
      }else{
	for(uint i=0; i<m; ++i)
	  std::copy(dcmp.col_begin(i),     // triangular part
		    dcmp.col_begin(i)+i+1,
		    ans.col_begin(i));
	for(uint i=m; i<n; ++i)
	  std::copy(dcmp.col_begin(i),     // rectangular part
		    dcmp.col_begin(i)+m,
		    ans.col_begin(i));
      }
      return ans;
    }

    Matrix QR::solve(const Matrix &B)const{
      Matrix ans(B);
      int m = dcmp.nrow();
      int n = dcmp.ncol(); // same as B.nrow
      int k = std::min(m,n);
      int ncol_b = ans.ncol();
      Vector work(1);
      int lwork = -1;
      int info=0;

      // set ans = Q^T*B
      dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
	      ans.data(), &n, work.data(), &lwork, &info);
      lwork = static_cast<int>(work[0]);
      work.resize(lwork);
      dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
	      ans.data(), &n, work.data(), &lwork, &info);

      // set ans = R^{-1} * ans
      dtrtrs_("U", "N", "N", &k, &ncol_b, dcmp.data(), &m,
	      ans.data(), &n, &info);
      return ans;
    }


    Vector QR::Qty(const Vector &y)const{
      Vector ans(y);
      const char * side="L";
      const char * trans="T";
      int m = y.size();
      int n = 1;
      int k = std::min(dcmp.nrow(),dcmp.ncol());
      const double * a=dcmp.data();
      Vector work(1);
      int lwork = -1;
      int info=0;

      // set ans = Q^T*y          comments show LAPACK argument names
      dormqr_(side,                // side
	      trans,                // trans
	      &m,                 // m   nrow(y)
	      &n,                 // n   ncol(y) := 1
	      &k,                 // k   dcmp.nrow()
	      a,                  // a
	      &m,                 // lda
	      tau.data(),         // tau
	      ans.data(),         // C
	      &m,                 // ldc
	      work.data(),        // work
	      &lwork,             // lwork
	      &info);             // info
      lwork = static_cast<int>(work[0]);
      work.resize(lwork);
      dormqr_(side,
	      trans,
	      &m,
	      &n,
	      &k,
	      a,
	      &m,
	      tau.data(),
	      ans.data(),
	      &m,
	      work.data(),
	      &lwork,
	      &info);
      return ans;
    }


    Matrix QR::QtY(const Matrix &Y)const{
      Matrix ans(Y);
      int m = ans.nrow();
      int n = ans.ncol();
      int k = std::min(m,n);
      int lda = dcmp.nrow();
      Vector work(1);
      int lwork = -1;
      int info=0;

      // set ans = Q^T*Y
      dormqr_("L", "T", &m, &n, &k, dcmp.data(), &lda, tau.data(),
	      ans.data(), &m, work.data(), &lwork, &info);
      lwork = static_cast<int>(work[0]);
      work.resize(lwork);
      dormqr_("L", "T", &m, &n, &k, dcmp.data(), &lda, tau.data(),
	      ans.data(), &m, work.data(), &lwork, &info);
      return ans;
    }

    Vector QR::solve(const Vector &B)const{
      Vector ans(B);
      int m = dcmp.nrow();
      int n = dcmp.ncol(); // same as B.nrow
      int k = std::min(m,n);
      int ncol_b = 1;
      Vector work(1);
      int lwork = -1;
      int info=0;

      // set ans = Q^T*B
      dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
	      ans.data(), &n, work.data(), &lwork, &info);
      lwork = static_cast<int>(work[0]);
      work.resize(lwork);
      dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
	      ans.data(), &n, work.data(), &lwork, &info);

      // set ans = R^{-1} * ans
      dtrtrs_("U", "N", "N", &k, &ncol_b, dcmp.data(), &m,
	      ans.data(), &n, &info);
      return ans;
    }

    double QR::det()const{
      double ans = 1.0;
      uint m = std::min(dcmp.nrow(), dcmp.ncol());
      for(uint i=0; i<m; ++i) ans*= dcmp.unchecked(i,i);
      return ans; }

    void QR::decompose(const Matrix &mat){
      dcmp = mat;
      tau.resize(std::min(dcmp.nrow(), dcmp.ncol()));
      work.resize(dcmp.ncol());
      int m = dcmp.nrow();
      int n = dcmp.ncol();
      int info=0;
      lwork = -1;
      // have LAPACK compute optimal lwork...
      dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
      lwork = static_cast<int>(work[0]);
      work.resize(lwork);
      // compute the decomposition with the optimal value...
      dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
    }

    void QR::clear(){
      dcmp = Matrix();
      tau = Vector();
      work = Vector();
      lwork = -1;
    }

    Vector QR::Rsolve(const Vector &Qty)const{
      Vector ans(Qty);
      Matrix R(getR());
      cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
		  R.nrow(), R.data(), R.nrow(), ans.data(), ans.stride());
      return ans;
    }

    Matrix QR::Rsolve(const Matrix & QtY)const{
      Matrix ans(QtY);
      Matrix R(getR());
      int m = ans.nrow();
      int n = ans.ncol();
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		  m,
		  n,
		  1.0,
		  R.data(),
		  R.nrow(),
		  ans.data(),
		  ans.nrow());
      return ans;
    }
  }
}
