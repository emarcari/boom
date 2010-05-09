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

#ifndef BOOM_MLVS_SPLIT_HPP
#define BOOM_MLVS_SPLIT_HPP

#include <Models/Glm/PosteriorSamplers/MLVS_base.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{

  class MvnBase;
  class VariableSelectionPrior;
  class MLogitSplit;


  class MlvsCdSuf_split : public MlvsCdSuf{
  public:
    MlvsCdSuf_split(uint dim, uint Nch);
    virtual void clear();
    virtual void update(Ptr<ChoiceData>, const Vec & wgts, const Vec &u);
    virtual MlvsCdSuf_split * clone()const;
    virtual void add(Ptr<MlvsCdSuf>);
    void add(Ptr<MlvsCdSuf_split>);
    const Spd & xtwx(uint which)const;
    const Vec & xtwu(uint which)const;
  private:
    mutable std::vector<Spd> xtwx_;
    std::vector<Vec> xtwu_;
    mutable bool sym_;
  };

  class MLVS_split : public MLVS_base {
    // This sampler assumes that a multinomial logit model is based on
    // all subject data (no choice level covariates), and that
    // independent conditionally MVN-VSP priors are assumed for the
    // coefficients of each choice level.

    typedef VariableSelectionPrior VSP;

  public:
    MLVS_split(Ptr<MLogitSplit>, std::vector<Ptr<MvnBase> >,
	       std::vector<Ptr<VSP> >, uint nthreads=1);
    MLVS_split(Ptr<MLogitSplit>, Ptr<MvnBase>, Ptr<VSP>, uint nthreads=1); // symmetric priors
    virtual double logpri()const;
    virtual void find_posterior_mode();
  private:
    Ptr<MLogitSplit> mod_;
    std::vector<Ptr<MvnBase> > bpri_;
    std::vector<Ptr<VSP> > vpri_;
    Ptr<MlvsCdSuf_split> suf;
    Ptr<MlvsDataImputer> imp;

    Spd Ominv;
    Spd iV_tilde_;
    Vec beta_tilde_;

    virtual void impute_latent_data();
    virtual void draw_beta();
    virtual void draw_inclusion_vector();

    double log_model_prob(const Selector &g, uint m);
    void setup();
  };
}
#endif// BOOM_MLVS_SPLIT_HPP
