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

#ifndef EXPONENTIAL_MODEL_H
#define EXPONENTIAL_MODEL_H
#include <iosfwd>
#include <cpputil/Ptr.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/DoubleModel.hpp>
#include <Models/EmMixtureComponent.hpp>
#include "Sufstat.hpp"
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>


//======================================================================
namespace BOOM{
  class ExpSuf: public SufstatDetails<DoubleData>{
    double sum_, n_;
  public:
    ExpSuf();
    ExpSuf(const ExpSuf &);
    ExpSuf *clone() const;

    void clear();
    void Update(const DoubleData &dat);
    void add_mixture_data(double y, double prob);
    double sum()const;
    double n()const;
    void combine(Ptr<ExpSuf>);
    void combine(const ExpSuf &);
    ExpSuf * abstract_combine(Sufstat *s);
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  };
  //======================================================================
  class GammaModel;
  class ExponentialGammaSampler;

  class ExponentialModel:
    public ParamPolicy_1<UnivParams>,
    public SufstatDataPolicy<DoubleData,ExpSuf>,
    public ConjugatePriorPolicy<ExponentialGammaSampler>,
    public DiffDoubleModel,
    public NumOptModel,
    public EmMixtureComponent
  {
  public:
    ExponentialModel();
    ExponentialModel(double lam);
    ExponentialModel(const ExponentialModel &m);
    ExponentialModel *clone() const;

    Ptr<UnivParams> Lam_prm();
    const Ptr<UnivParams> Lam_prm()const;
    const double& lam() const;
    void set_lam(double);

    void set_conjugate_prior(double a, double b);
    void set_conjugate_prior(Ptr<GammaModel>);
    void set_conjugate_prior(Ptr<ExponentialGammaSampler>);

    // probability calculations
    virtual double pdf(Ptr<Data> dp, bool logscale)const;
    virtual double pdf(const Data * dp, bool logscale)const;
    double Loglike(Vec &g, Mat &h, uint lev) const ;
    double Logp(double x, double &g, double &h, const uint lev) const ;
    double sim() const;
    void add_mixture_data(Ptr<Data>, double prob);
  };



}
#endif  // EXPONENTIALMODEL_H
