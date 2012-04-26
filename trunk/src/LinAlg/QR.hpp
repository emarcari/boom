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
#ifndef BOOM_NEWLA_QR_HPP
#define BOOM_NEWLA_QR_HPP

#include "Matrix.hpp"
#include "Vector.hpp"

namespace BOOM{
  class QR{
    Matrix dcmp;
    Vector tau;
    Vector work;  // work space
    int lwork;
   public:
    QR(){}
    QR(const Matrix &m);
    Matrix getQ()const;
    Matrix getR()const;
    Matrix solve(const Matrix &B)const;
    Vector solve(const Vector &b)const;
    Vector Qty(const Vector &y)const;
    Matrix QtY(const Matrix &Y)const;
    Vector Rsolve(const Vector &Qty)const;
    Matrix Rsolve(const Matrix &QtY)const;
    double det()const;
    void decompose(const Matrix &m);
    void clear();
    uint nrow()const{return dcmp.nrow();}
    uint ncol()const{return dcmp.ncol();}
  };
}
#endif //BOOM_NEWLA_QR_HPP
