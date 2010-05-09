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
#include "Array4.hpp"
#include <iosfwd>
#include <iomanip>

namespace BOOM{
  namespace LinAlg{

    Array4::Array4()
      : V(new a3vector){}
    Array4::Array4(uint d1, uint d2, uint d3, uint d4, double x)
      : V(new a3vector(d1, Array3(d2,d3,d4,x))),
	n1(d1), n2(d2),n3(d3),n4(d4)
    {}

    Array4::Array4(uint dim, double x)
      : V(new a3vector(dim, Array3(dim,dim,dim,x))),
	n1(dim),
	n2(dim),
	n3(dim),
	n4(dim)
    {}

    Array4::Array4(const Array4 &rhs)
      : V(rhs.V),
	n1(rhs.n1),
	n2(rhs.n2),
	n3(rhs.n3),
	n4(rhs.n4)
    {}

    Array4 & Array4::operator=(const Array4 &rhs){
      if(&rhs!=this){
	n1 = rhs.n1;
	n2 = rhs.n2;
	n3 = rhs.n3;
	n4 = rhs.n4;
	V->resize(n1);
	std::copy(rhs.V->begin(), rhs.V->end(), V->begin());
      }
      return *this;
    }

    Array4 & Array4::operator=(const double &x){
      if(n1==0){
	n1 = n2 = n3 = n4 = 1;
	V->resize(1);
	(*V)[0] = Array3(n2,n3,n4, x);
      }else for(uint i=0; i<n1; ++i) (*V)[i] = x;
      return *this;
    }


    std::ostream & operator<<(std::ostream &out, Array4 &x){
      out << std::setprecision(5);
      for(uint i = x.lo1(); i<=x.hi1(); ++i){
 	for(uint j = x.lo2(); j<=x.hi2(); ++j){
 	  for(uint k = x.lo3(); k<=x.hi3(); ++k){
 	    for(uint m = x.lo4(); m<=x.hi4(); ++m){
 	      out << x(i,j,k, m) << " ";
 	    }
 	    out << std::endl;  // end of line
 	  }
 	  out << std::endl;  // extra end of line after each block
 	}
 	out << std::endl; // two line breaks after each outside block
      }
      return out;
    }



  }
}
