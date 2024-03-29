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

#ifndef BOOM_SPD_PARAMS_HPP
#define BOOM_SPD_PARAMS_HPP
#include "SpdData.hpp"
#include "ParamTypes.hpp"

namespace BOOM{
  /*
   * There are several storage classes that one could use for Spd
   * params.  Store Sigma, Sigma_inverse, chol(Sigma), chol(Sigma_inv),
   * S and R, S and chol(R), S and chol(R_inv), etc.
   */

  class SpdParams
    : public SpdData,
      virtual public Params
  {
  public:
    explicit SpdParams(uint p, double diag = 1.0, bool ivar=false);
    SpdParams(const Spd &V, bool ivar);
    SpdParams(const SpdParams &rhs);
    SpdParams(const SpdData &rhs);
    SpdParams * clone()const;

    virtual uint size(bool minimal = true)const{
      return SpdData::size(minimal);}
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
 					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
 					    bool minimal=true);

  };



}
#endif// BOOM_SPD_PARAMS_HPP
