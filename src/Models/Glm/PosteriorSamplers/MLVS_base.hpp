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

#ifndef BOOM_MLVS_BASE_HPP
#define BOOM_MLVS_BASE_HPP

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Vector.hpp>
#include <BOOM.hpp>

namespace BOOM{

  class MLogitBase;
  class MvnBase;
  class ChoiceData;
  class Selector;

  // Abstract base class for multinomial logit complete data
  // sufficient statistics.  The plan is to have a version of this
  // class for both ordinary multinomial logit models and the split
  // parameter version.
  class MlvsCdSuf : private RefCounted {
  public:
    friend void intrusive_ptr_add_ref(MlvsCdSuf *d){d->up_count();}
    friend void intrusive_ptr_release(MlvsCdSuf *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

    virtual ~MlvsCdSuf(){}
    virtual MlvsCdSuf * clone()const=0;

    virtual void clear()=0;
    virtual void update(Ptr<ChoiceData> dp, const Vec &wgts, const Vec & u)=0;

    virtual void add(Ptr<MlvsCdSuf>)=0;
  };

  class MLVS_base
    : virtual public PosteriorSampler
  {
    // Base class for drawing the parameters of a multinomial logit
    // model using the approximate method from Fruewirth-Schnatter and
    // Fruewirth, Computational Statistics and Data Analysis 2007,
    // 3508-3528.

    // this implementation only stores the complete data sufficient
    // statistics and some workspace.  It does not store the imputed
    // latent data.
  public:
    MLVS_base(MLogitBase *, bool do_selection = true);

    virtual void draw();
    virtual void impute_latent_data()=0;
    virtual void draw_inclusion_vector()=0;
    virtual void draw_beta()=0;
    //    virtual void find_posterior_mode()=0;
    // find_posterior_mode does not search over model space.  The
    // current model (which x's are in/out) is held fixed.

    void supress_model_selection();
    void allow_model_selection();
    void supress_beta_draw();
    void allow_beta_draw();

    void limit_model_selection(uint nflips);
    uint max_nflips()const;

  private:
    const Vec & log_sampling_probs_;
    const bool downsampling_;
    bool select_;
    bool beta_draw_;
    uint max_nflips_;
  };
}
#endif// BOOM_MLVS_BASE_HPP
