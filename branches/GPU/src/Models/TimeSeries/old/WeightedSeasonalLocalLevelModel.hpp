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

#ifndef BOOM_WEIGHTED_SEASONAL_LOCAL_LEVEL_MODEL_HPP
#define BOOM_WEIGHTED_SEASONAL_LOCAL_LEVEL_MODEL_HPP

#include "SeasonalLocalLevelModel.hpp"

namespace BOOM{


  class weighted_seasonal_local_level_suf
    : public seasonal_local_level_suf{
  public:
    weighted_seasonal_local_level_suf();
    weighted_seasonal_local_level_suf(const weighted_seasonal_local_level_suf &rhs);
    weighted_seasonal_local_level_suf * clone()const;

    void Update(const StateSpaceData &d);
  };

  class WeightedSeasonalLocalLevelModel
    : public SeasonalLocalLevelModel
  {
  public:
    typedef WeightedSeasonalLocalLevelModel WSLLM;

    WeightedSeasonalLocalLevelModel();
    WeightedSeasonalLocalLevelModel(uint NSeasons);
    WeightedSeasonalLocalLevelModel(double Siglev, double SigSeason, double SigObs,
					uint NSeasons);
    WeightedSeasonalLocalLevelModel(const std::vector<double> &dv, uint NSeasons);
    WeightedSeasonalLocalLevelModel(const WSLLM &);
    WSLLM * clone()const;

    virtual Spd residual_variance(Ptr<ssd>)const;        // H_t
    virtual Spd Sigma()const;

  };
}
#endif // BOOM_WEIGHTED_SEASONAL_LOCAL_LEVEL_MODEL_HPP
