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

#ifndef BOOM_FINITE_MIXTURE_MODEL_HPP
#define BOOM_FINITE_MIXTURE_MODEL_HPP

#include <Models/ModelTypes.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/MixtureDataPolicy.hpp>
#include <Models/MultinomialModel.hpp>

namespace BOOM{

  class FiniteMixtureModel
    : public LoglikeModel,
      public LatentVariableModel,
      public CompositeParamPolicy,
      public MixtureDataPolicy
  {
  public:
    FiniteMixtureModel(Ptr<LoglikeModel>, uint S);
    FiniteMixtureModel(Ptr<LoglikeModel>, Ptr<MultinomialModel>);

    template <class M>
    FiniteMixtureModel(std::vector<Ptr<M> >, Ptr<MultinomialModel>);

    template <class FwdIt>
    FiniteMixtureModel(FwdIt Beg, FwdIt End, Ptr<MultinomialModel>);

    FiniteMixtureModel(const FiniteMixtureModel &rhs);
    FiniteMixtureModel * clone()const;

    virtual double loglike()const;
    virtual void mle();
    void clear_component_data();
    virtual void impute_latent_data();
    virtual double complete_data_loglike()const;
    void EStep();
    void MStep();

    virtual double logpri()const;
    virtual void sample_posterior();
    virtual void set_method(Ptr<PosteriorSampler>);

    double pdf(dPtr dp, bool logscale)const;
    uint size()const;

    const Vec & pi()const;

    Ptr<MultinomialModel> mixing_distribution();
    const Ptr<MultinomialModel> mixing_distribution()const;

    std::vector<Ptr<LoglikeModel> > mixture_components();
    const std::vector<Ptr<LoglikeModel> > mixture_components()const;

  private:
    std::vector<Ptr<LoglikeModel> > mixture_components_;
    Ptr<MultinomialModel> mixing_dist_;
    mutable Vec logpi_, wsp_;
    mutable bool logpi_current_;
    void observe_pi()const;
    void set_logpi()const;
    void set_observers();
    virtual std::vector<Ptr<LoglikeModel> > models();
    virtual const std::vector<Ptr<LoglikeModel> > models()const;
  };


  template <class FwdIt>
  FiniteMixtureModel::FiniteMixtureModel(FwdIt Beg, FwdIt End,
                                         Ptr<MultinomialModel> MixDist)
    : DataPolicy(MixDist->size()),
      mixture_components_(Beg,End),
      mixing_dist_(MixDist)
  {
    set_observers();
  }

  template <class M>
  FiniteMixtureModel::FiniteMixtureModel(std::vector<Ptr<M> > Models,
                                         Ptr<MultinomialModel> MixDist)
    : DataPolicy(MixDist->size()),
      mixture_components_(Models.begin(), Models.end()),
      mixing_dist_(MixDist)
  {
    set_observers();
  }

}
#endif// BOOM_FINITE_MIXTURE_MODEL_HPP
