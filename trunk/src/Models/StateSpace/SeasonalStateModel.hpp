/*
  Copyright (C) 2008 Steven L. Scott

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

#ifndef BOOM_SEASONALSTATE_MODEL_HPP
#define BOOM_SEASONALSTATE_MODEL_HPP

namespace BOOM{
  
  class SeasonalStateModel
    : public HomogeneousStateModel, 
      public ParamPolicy_1<UnivParams>
  {
  public:
    
    double sigsq()const;
    void set_sigsq(double);
    Ptr<UnivParams> Sigsq_prm();

    virtual observe_state(const ConstVectorView & now, const ConstVectorView & next)=0;

  private:
    Ptr<GaussianSuf> suf_;
    uint nseasons_;

    // state is (s[t], s[t-1], ... s[t-nseasons_])  ...
    // contribution to y[t] is s[t] (i.e. Z = (1,0,0,0,...)  )
    
  };

}

#endif // BOOM_SEASONALSTATE_MODEL_HPP
