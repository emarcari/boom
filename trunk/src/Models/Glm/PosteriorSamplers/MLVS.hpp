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

#ifndef BOOM_MULTINOMIAL_LOGIT_VARIABLE_SELECTION_HPP
#define BOOM_MULTINOMIAL_LOGIT_VARIABLE_SELECTION_HPP


#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_base.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp>


namespace BOOM{

  class MultinomialLogitModel;
  class MvnBase;
  class ChoiceData;

  // An implementation of
  class MlvsCdSuf_ml : public MlvsCdSuf{
  public:
    MlvsCdSuf_ml(uint dim);

    MlvsCdSuf_ml(const Spd & inMatrix, const Vec & inVector);

    virtual MlvsCdSuf_ml * clone()const;
    virtual void clear();
    virtual void update(const Ptr<ChoiceData> dp, const Vec &wgts,
			const Vec &u);
    void update(const Spd & mat, const Vec & weightedU);
    virtual void add(Ptr<MlvsCdSuf>);
    void add(Ptr<MlvsCdSuf_ml>);

    const Spd & xtwx()const;
    const Vec & xtwu()const;

  private:
    mutable Spd xtwx_;
    Vec xtwu_;
    mutable bool sym_;
  };

  //------------------------------------------------------------
  class MLVS : public MLVS_base{
    // draws the parameters of a multinomial logit model using the
    // approximate method from Fruewirth-Schnatter and Fruewirth,
    // Computational Statistics and Data Analysis 2007, 3508-3528.

    // this implementation only stores the complete data sufficient
    // statistics and some workspace.  It does not store the imputed
    // latent data.
  public:
    MLVS(MultinomialLogitModel *Mod,
	 Ptr<MvnBase> Pri,
	 Ptr<VariableSelectionPrior> vPri,
	 uint nthreads=1,
	 bool check_initial_condition=true,
	 int mode=0);
    virtual double logpri()const;
    virtual void impute_latent_data();
    virtual void draw_beta();
    //    virtual void find_posterior_mode();
  private:
    MultinomialLogitModel *mod_;
    Ptr<MvnBase> pri;
    Ptr<VariableSelectionPrior> vpri;
    Ptr<MlvsCdSuf_ml> suf;
    Ptr<MlvsDataImputer> imp;

    Spd Ominv;
    Spd iV_tilde_;
    virtual void draw_inclusion_vector();
    double log_model_prob(const Selector &inc);
  };

  //------------------------------------------------------------


}
#endif//  BOOM_MULTINOMIAL_LOGIT_VARIABLE_SELECTION_HPP
