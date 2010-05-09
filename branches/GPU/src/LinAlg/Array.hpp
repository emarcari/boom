/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_ARRAY_GEN_HPP
#define BOOM_ARRAY_GEN_HPP

namespace BOOM{
  namespace LinAlg{

    template <unsigned N>
    class Array{
    public:
      Array(std::vector<uint> );
      std::vector<uint> dims()const;

      const Array<N-1>
    private:
      std::vector<Array<N-1> > v_;
    };

    template <>
    class Array<1>{
    public:

    private:
      Vector v_;
    };

    template<>
    class Array<2>{
    public:

    private:
      Matrix v_;
    };

  }
}
#endif// BOOM_ARRAY_GEN_HPP
