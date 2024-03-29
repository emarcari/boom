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

#ifndef NEW_LA_SPD_MATRIX_H
#define NEW_LA_SPD_MATRIX_H
#include "Matrix.hpp"

namespace BOOM{
  using std::ostream;
  using std::istream;

  class Vector;
  class Matrix;

  class SpdMatrix
    : public Matrix
  {
    // symmetric, positive definite mattrix with 'square' storage
    // (i.e. 0's are stored)
  public:
    SpdMatrix();
    SpdMatrix(uint dim, double diag=0.0);
    SpdMatrix(uint dim, double *m, bool ColMajor=true);
    template <class FwdIt>
    explicit SpdMatrix(FwdIt Beg, FwdIt End);
    SpdMatrix(const SpdMatrix &sm);  // reference semantics
    SpdMatrix(const Matrix &m, bool check=true);
    SpdMatrix(const SubMatrix &m, bool check=true);

    SpdMatrix & operator=(const SpdMatrix &); // value semantics
    SpdMatrix & operator=(const Matrix &);
    SpdMatrix & operator=(const SubMatrix &);
    SpdMatrix & operator=(double x);
    bool operator==(const SpdMatrix &)const;

    void  swap(SpdMatrix &rhs);
    void randomize();  // fills entries with U(0,1) random variables,
                       // then multiply by self-transpose.

    //-------- size and shape info ----------
    virtual uint nelem()const;         // number of distinct elements
    uint dim()const{return nrow();}

    //--------- change size and shape ----------
    SpdMatrix & resize(uint n);

    // -------- row and column operations ----------
    SpdMatrix & set_diag(double x, bool zero_offdiag=true);
    SpdMatrix & set_diag(const Vector &v, bool zero_offdiag=true);

    //------------- Linear Algebra -----------
    //      lower_triangular_Matrix chol() const;
    Matrix chol() const;
    Matrix chol(bool & ok) const;
    SpdMatrix inv()const;
    SpdMatrix inv(bool &ok)const;
    double det() const;
    double logdet() const;
    double logdet(bool &ok) const;
    Matrix solve(const Matrix &mat) const;
    Vector solve(const Vector &v) const;

    void reflect();   // copies upper triangle into lower triangle
    double Mdist(const Vector &x, const Vector &y) const ;
    double Mdist(const Vector &x) const ;

    SpdMatrix & add_outer(const Vector &x, double w = 1.0,
      		    bool force_sym=true);     // *this+= w*x*x^T
    SpdMatrix & add_outer(const VectorView &x, double w = 1.0,
      		    bool force_sym=true);     // *this+= w*x*x^T
    SpdMatrix & add_outer(const ConstVectorView &x, double w = 1.0,
      		    bool force_sym=true);     // *this+= w*x*x^T
    SpdMatrix & add_outer(const Matrix &X, double w=1.0,
                          bool force_sym = true);   // *this+= w*X*X^T

    SpdMatrix & add_outer_w(const Vector &x, double w = 1.0){
      return add_outer(x,w); }

    SpdMatrix & add_inner(const Matrix &x, double w=1.0);
    SpdMatrix & add_inner(const Matrix &X, const Vector & w,
      		    bool force_sym=true);  // *this+= X^T w X

    // *this  += w x.t()*y + y.t()*x;
    SpdMatrix & add_inner2(const Matrix &x, const Matrix &y, double w=1.0);
    // *this  += w x*y.t() + y*x.t();
    SpdMatrix & add_outer2(const Matrix &x, const Matrix &y, double w=1.0);

    SpdMatrix & add_outer2(const Vector &x, const Vector &y, double w = 1.0);

    //--------- Matrix multiplication ------------
    Matrix & mult(const Matrix &B, Matrix &ans, double scal=1.0)const;
    Matrix & Tmult(const Matrix &B, Matrix &ans, double scal=1.0)const;
    Matrix & multT(const Matrix &B, Matrix &ans, double scal=1.0)const;

    Matrix & mult(const SpdMatrix &B, Matrix &ans, double scal=1.0)const;
    Matrix & Tmult(const SpdMatrix &B, Matrix &ans, double scal=1.0)const;
    Matrix & multT(const SpdMatrix &B, Matrix &ans, double scal=1.0)const;

    Matrix & mult(const DiagonalMatrix &B, Matrix &ans, double scal=1.0)const;
    Matrix & Tmult(const DiagonalMatrix &B, Matrix &ans, double scal=1.0)const;
    Matrix & multT(const DiagonalMatrix &B, Matrix &ans, double scal=1.0)const;

    Vector & mult(const Vector &v, Vector &ans, double scal=1.0)const;
    Vector & Tmult(const Vector &v, Vector &ans, double scal=1.0)const;

    //------------- input/output ---------------
    virtual Vector vectorize(bool minimal=true)const;
    virtual void unvectorize(const Vector &v, bool minimal=true);
    Vector::const_iterator unvectorize(Vector::const_iterator &b,
      				 bool minimal=true);
    void make_symmetric(bool have_upper_triangle=true);
  };

  typedef SpdMatrix Spd;

  //______________________________________________________________________
  template <class Fwd>
  SpdMatrix::SpdMatrix(Fwd b, Fwd e){
    uint n = std::distance(b,e);
    uint m = lround( ::sqrt(n));
    assert(m*m == n);
    resize(m);
    std::copy(b,e,begin());
  }

  SpdMatrix operator*(double x, const SpdMatrix &V);
  SpdMatrix operator*(const SpdMatrix &v, double x);
  SpdMatrix operator/(const SpdMatrix &v, double x);

  SpdMatrix Id(uint p);

  SpdMatrix RTR(const Matrix &R, double a = 1.0); // a * R^T%*%R
  SpdMatrix LLT(const Matrix &L, double a = 1.0); // a * L%*%L^T

  SpdMatrix outer(const Vector &v);
  SpdMatrix outer(const VectorView &v);
  SpdMatrix outer(const ConstVectorView &v);

  Matrix chol(const SpdMatrix &Sigma);
  Matrix chol(const SpdMatrix &Sigma, bool &ok);

  inline double logdet(const SpdMatrix & Sigma){ return Sigma.logdet();}

  SpdMatrix chol2inv(const Matrix &L);
  // Returns A^{-1}, where L is the cholesky factor of A.

  SpdMatrix sandwich(const Matrix &A, const SpdMatrix &V); // AVA^t
  SpdMatrix sandwich_old(const Matrix &A, const SpdMatrix &V); // AVA^t

  SpdMatrix select(const SpdMatrix &X, const std::vector<bool> &inc,
      	      uint nvars);
  SpdMatrix select(const SpdMatrix &X, const std::vector<bool> &inc);
  SpdMatrix as_symmetric(const Matrix &A);

  SpdMatrix sum_self_transpose(const Matrix &A);  // A + A.t()

  Vector eigen(const SpdMatrix &X);
  Vector eigen(const SpdMatrix &X, Matrix & evec);
}
#endif // NEW_LA_SPD_MATRIX_H
