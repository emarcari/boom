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
#ifndef BOOM_PARTIALLY_OBSERVED_MVN_MODEL_HPP
#define BOOM_PARTIALLY_OBSERVED_MVN_MODEL_HPP

namespace BOOM{

  class PoMvnSuf : public SufstatDetails<GlmCoefs>{
  public:
    PoMvnSuf(uint dim);
    PoMvnSuf(const PoMvnSuf&);
    PoMvnSuf * clone()const;

    void clear();
    void Update(const GlmCoefs &);

    const Vec & sum()const;
    const Spd & sumsq()const;
    double n()const;

  private:
    Vec sum_;     // sum( gamma(beta))
    Spd sumsq_;   // sum(gamma(beta) * gamma(beta).t())
    double n_;    //
  };

}
#endif // BOOM_PARTIALLY_OBSERVED_MVN_MODEL_HPP
