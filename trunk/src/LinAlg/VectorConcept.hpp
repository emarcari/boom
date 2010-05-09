/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_VECTOR_CONCEPT_HPP
#define BOOM_VECTOR_CONCEPT_HPP

#include <boost/concept_check.hpp>

namespace BOOM{
  namespace LinAlg{


    template <class VEC>
    struct VectorConcept{
      void constraints(){
	v.begin();
	v.end();
	x = v[i];
      }
      double x;
      unsigned int i;
      VEC v;
    };
  }
}
#endif// BOOM_VECTOR_CONCEPT_HPP
