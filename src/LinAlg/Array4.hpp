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

#ifndef BOOM_NEWLA_ARRAY4_HPP
#define BOOM_NEWLA_ARRAY4_HPP

#include <boost/shared_ptr.hpp>
#include <vector>
#include "Array3.hpp"
#include <iosfwd>

namespace BOOM{
  namespace LinAlg{
    class Array4{
      typedef std::vector<Array3> a3vector;
      boost::shared_ptr<a3vector> V;
      uint n1, n2, n3, n4;
    public:
      Array4();
      Array4(uint d1, uint d2, uint d3, uint d4, double x=0.0);
      Array4(uint dim, double x=0.0);
      Array4(const Array4 &a);

      Array4 & operator=(const Array4 &rhs);
      Array4 & operator=(const double &x);

      uint lo1()const{ return 0;}
      uint lo2()const{ return 0;}
      uint lo3()const{ return 0;}
      uint lo4()const{ return 0;}

      uint hi1() const {return n1-1;}
      uint hi2() const {return n2-1;}
      uint hi3() const {return n3-1;}
      uint hi4() const {return n4-1;}

      inline Array3 & operator[](uint i);
      inline const Array3 & operator[](uint i)const;
      inline Matrix & operator()(uint i, uint j);
      inline const Matrix & operator()(uint i, uint j) const;

      inline double &  operator()(uint i, uint j, uint k, uint m);
      inline const double &operator()(uint i, uint j, uint k, uint m) const;

    };

    std::ostream & operator<<(std::ostream & out, const Array4 &x);

    //----------inlines
    inline Array3 & Array4::operator[](uint i){
      return (*V)[i];}
    inline const Array3 & Array4::operator[](uint i)const{
      return (*V)[i];}

    inline Matrix & Array4::operator()(uint i, uint j){
      return (*V)[i][j];}
    inline const Matrix & Array4::operator()(uint i, uint j)const{
      return (*V)[i][j];}

    inline double &
    Array4::operator()(uint i, uint j, uint k, uint m){
      return (*V)[i](j,k,m); }
    inline const double &
    Array4::operator()(uint i, uint j, uint k, uint m)const{
      return (*V)[i](j,k,m); }



  }
}
#endif // BOOM_NEWLA_ARRAY4_HPP
