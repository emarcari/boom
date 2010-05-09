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
#ifndef BOOM_LIN_ALG_TYPES_HPP
#define BOOM_LIN_ALG_TYPES_HPP

namespace BOOM{

  namespace LinAlg{    // BOOM_USE_ATLAS is defined
    class Vector;
    class VectorView;
    class ConstVectorView;
    class Matrix;
    class SpdMatrix;
    class CorrelationMatrix;
    class Array3;
    class Array4;
    class QR;
    class Chol;
  }
  struct LinAlgTypes{
    typedef LinAlg::Vector Vec;
    typedef LinAlg::VectorView VectorView;
    typedef LinAlg::ConstVectorView ConstVectorView;
    typedef LinAlg::Matrix Mat;
    typedef LinAlg::SpdMatrix Spd ;
    typedef LinAlg::CorrelationMatrix Corr ;
    typedef LinAlg::Array3 Arr3;
    typedef LinAlg::Array4 Arr4;
    typedef LinAlg::QR QR;
    typedef LinAlg::Chol Chol;
    typedef unsigned int Int;
  };

  typedef LinAlg::Vector Vec;
  typedef LinAlg::VectorView VectorView;
  typedef LinAlg::ConstVectorView ConstVectorView;
  typedef LinAlg::Matrix Mat;
  typedef LinAlg::SpdMatrix Spd;
  typedef LinAlg::Chol Chol;
  typedef LinAlg::CorrelationMatrix Corr;
  typedef LinAlg::Array3 Arr3;
  typedef LinAlg::Array4 Arr4;

}
#endif // BOOM_LIN_ALG_TYPES_HPP
