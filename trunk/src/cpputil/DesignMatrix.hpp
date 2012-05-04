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

#ifndef BOOM_DESIGN_MATRIX_HPP
#define BOOM_DESIGN_MATRIX_HPP
#include <BOOM.hpp>

#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Types.hpp>
#include <limits>

namespace BOOM{
  class DesignMatrix : public Mat
  {
    // A design matrix is just a matrix with variable names.
  public:
    DesignMatrix(const Mat &X);
    DesignMatrix(const Mat &X, const std::vector<string> &vnames);
    DesignMatrix(const Mat &X, const std::vector<string> &vnames,
 		  const std::vector<string> &baseline_names);
    DesignMatrix(const DesignMatrix &);

    std::vector<string> &vnames(){return vnames_;}
    const std::vector<string> &vnames()const {return vnames_;}
  private:
    std::vector<string> vnames_;
    std::vector<string> baseline_names_;
  };

  ostream & display(ostream &out, const DesignMatrix &m,
		    int prec=5, uint from=0,
		    uint to= std::numeric_limits<uint>::max());
  ostream & operator<<(ostream &out, const DesignMatrix &m);
}
#endif// BOOM_DESIGN_MATRIX_HPP
