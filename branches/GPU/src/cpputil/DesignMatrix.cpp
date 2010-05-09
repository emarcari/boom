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
#include "DesignMatrix.hpp"
#include "DefaultVnames.hpp"
#include <BOOM.hpp>
#include <iostream>
#include <iomanip>

namespace BOOM{
  DesignMatrix::DesignMatrix(const Mat &X)
    : Mat(X)
  {
    vnames_ = default_vnames(X.ncol());
  }

  DesignMatrix::DesignMatrix
  (const Mat &X, const std::vector<string> &vnames)
    : Mat(X),
      vnames_(vnames)
  { }

  DesignMatrix::DesignMatrix
  (const Mat &X, const std::vector<string> &vnames,
   const std::vector<string> &baseline_names)
    : Mat(X),
      vnames_(vnames),
      baseline_names_(baseline_names)
  {}

  DesignMatrix::DesignMatrix(const DesignMatrix &rhs)
    : Mat(rhs),
      vnames_(rhs.vnames_),
      baseline_names_(rhs.baseline_names_)
  {}



  //-----------------------------------------------------------------
  ostream & display(ostream & out, const DesignMatrix &m,
		    int prec, uint from, uint to){
    using std::setw;
    using std::setprecision;

    out << setprecision(prec);
    const std::vector<string> &vn(m.vnames());
    uint nvars = vn.size();
    std::vector<uint> fw(nvars);

    uint padding = 2;
    for(uint i=0; i<nvars; ++i){
      fw[i] = std::max<uint>(8u, vn[i].size()+padding);
    }

    for(uint i=0; i<m.vnames().size(); ++i)
      out << setw(fw[i]) << vn[i] ;
    out << endl;

    if(to>m.nrow()) to = m.nrow();
    for(uint i=from; i<to; ++i){
      for(uint j =0; j<m.ncol(); ++j){
	out << setw(fw[j]) << m(i,j);
      }
      out << endl;
    }
    return out;
  }

  //-----------------------------------------------------------------
  ostream & operator<<(ostream &out, const DesignMatrix &dm){
    return display(out, dm);  }
  //-----------------------------------------------------------------

}
