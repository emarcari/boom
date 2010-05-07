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

#include <Models/Glm/MLogitBase.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_base.hpp>
#include <cpputil/math_utils.hpp>  // for lse
#include <cpputil/lse.hpp>
#include <distributions.hpp>       // for rlexp,dnorm,rmvn

namespace BOOM{
  typedef MLVS_base MLVSB;

  MLVSB::MLVS_base(Ptr<MLogitBase> mod, bool do_select)
    : log_sampling_probs_(mod->log_sampling_probs()),
      downsampling_ (log_sampling_probs_.size() == mod->Nchoices()),
      select_(do_select),
      beta_draw_(true),
      max_nflips_(mod->beta_size(false))
  {}

  void MLVSB::supress_model_selection(){ select_ = false;}
  void MLVSB::allow_model_selection(){ select_ = true;}
  void MLVSB::supress_beta_draw(){ beta_draw_ = false; }
  void MLVSB::allow_beta_draw(){ beta_draw_ = true; }
  void MLVSB::limit_model_selection(uint n){max_nflips_ = n;}
  uint MLVSB::max_nflips()const{return max_nflips_;}
  void MLVSB::draw(){
    impute_latent_data();
    if(select_) draw_inclusion_vector();
    if(beta_draw_) draw_beta();
  }

}
