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
#ifndef BOOM_SUBMATRIX_HPP
#define BOOM_SUBMATRIX_HPP
#include <LinAlg/Matrix.hpp>
namespace BOOM{
  namespace LinAlg{

    // maintains a rectangular view into a matrix specified by lower
    // and upper coordinates (inclusive)
    class SubMatrix{
    public:
      typedef double * col_iterator;
      typedef const double * const_col_iterator;

      SubMatrix(Matrix &, uint rlo, uint rhi, uint clo, uint chi);
      SubMatrix(const SubMatrix &rhs);
      SubMatrix & operator=(const Matrix &rhs);
      SubMatrix & operator=(const SubMatrix &rhs);

      uint nrow()const;
      uint ncol()const;

      double & operator()(uint i, uint j);
      const double & operator()(uint i, uint j)const;

      col_iterator col_begin(uint j);
      const_col_iterator col_begin(uint j)const;

      col_iterator col_end(uint j);
      const_col_iterator col_end(uint j)const;

      VectorView col(uint j);
      ConstVectorView col(uint j)const;

      VectorView row(uint j);
      ConstVectorView row(uint j)const;

      SubMatrix & operator+=(const Matrix &m);
      SubMatrix & operator-=(const Matrix &m);

      double sum()const;

    private:
      std::vector<double *> cols_;
      uint nr_, nc_;
      uint stride;
    };
    //======================================================================
    class ConstSubMatrix{
    public:
      typedef const double * const_col_iterator;

      ConstSubMatrix(const Matrix &, uint rlo, uint rhi, uint clo, uint chi);

      uint nrow()const;
      uint ncol()const;

      const double & operator()(uint i, uint j)const;
      const_col_iterator col_begin(uint j)const;
      const_col_iterator col_end(uint j)const;

      ConstVectorView col(uint j)const;
      ConstVectorView row(uint j)const;

      double sum()const;
    private:
      std::vector<const double *> cols_;
      uint nr_, nc_;
      uint stride;
    };
  }
}
#endif // BOOM_SUBMATRIX_HPP
