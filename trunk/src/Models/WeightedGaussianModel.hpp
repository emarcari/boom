/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_WEIGHTED_GAUSSIAN_MODEL_H
#define BOOM_WEIGHTED_GAUSSIAN_MODEL_H

#include "ModelTypes.hpp"
#include "ParamTypes.hpp"
#include "Sufstat.hpp"
#include "Policies/SufstatDataPolicy.hpp"
#include "Policies/PriorPolicy.hpp"
#include "Policies/ParamPolicy_2.hpp"



namespace BOOM{

  class WeightedGaussianSuf
    : public SufstatDetails<WeightedDoubleData>{
  public:
    // constructor
    WeightedGaussianSuf();
    WeightedGaussianSuf(const WeightedGaussianSuf &);
    WeightedGaussianSuf *clone() const;

    void clear();
    void Update(const WeightedDoubleData &X);
    double sum()const;
    double sumsq()const;
    double n()const;
    double sumw()const;
    double sumlogw()const;

    double ybar()const;
    double sample_var()const;
    void combine(Ptr<WeightedGaussianSuf>);
  private:
    double sum_, sumsq_, n_, sumw_, sumlogw_;
  };

  //----------------------------------------------------------------------

  class WeightedGaussianModel
    : public ParamPolicy_2<UnivParams, UnivParams>,
      public PriorPolicy,
      public SufstatDataPolicy<WeightedDoubleData, WeightedGaussianSuf>,
      public DiffDoubleModel,
      public NumOptModel
  {


  GaussianModel();  // N(0,1)
    GaussianModel(double mean, double sd);
    GaussianModel(const std::vector<double> &v);
    GaussianModel(const GaussianModel &rhs);
    GaussianModel * clone()const;

    void set_params(double Mean, double Var);
    void set_mu(double m);
    void set_sigsq(double s);

    double ybar()const;
    double sample_var()const;

    double mu()const;
    double sigsq()const;
    double sigma()const;

    Ptr<UnivParams> Mu_prm();
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Mu_prm()const;
    const Ptr<UnivParams> Sigsq_prm()const;

    virtual void mle();

//     double pdf(Ptr<Data> dp, bool logscale)const;
//     double pdf(double x, bool logscale)const;
    double Logp(double x, double &g, double &h, uint nd)const;
    double Logp(const Vec & x, Vec &g, Mat &h, uint nd)const;
    double Loglike(Vec &g, Mat &h, uint nd)const;

}

#endif BOOM_WEIGHTED_GAUSSIAN_MODEL_H
