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
#ifndef BOOM_BETA_MODEL_HPP
#define BOOM_BETA_MODEL_HPP

#include "ModelTypes.hpp"
#include <Models/DoubleModel.hpp>
#include "ParamTypes.hpp"
#include "Sufstat.hpp"
#include "Policies/SufstatDataPolicy.hpp"
#include "Policies/PriorPolicy.hpp"
#include "Policies/ParamPolicy_2.hpp"

namespace BOOM{
  class BetaSuf: public SufstatDetails<DoubleData>{
  public:
    BetaSuf();
    BetaSuf(const BetaSuf &rhs);
    BetaSuf *clone() const;
    void clear(){n_=sumlog_ = sumlogc_ = 0.0;}
    void Update(const DoubleData &);
    void update_raw(double theta);
    double n()const{return n_;}
    double sumlog()const{return sumlog_;}
    double sumlogc()const{return sumlogc_;}
    BetaSuf * abstract_combine(Sufstat *s);
    void combine(Ptr<BetaSuf> s);
    void combine(const BetaSuf & s);
    virtual ostream &print(ostream &out)const;

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
   private:
    double n_, sumlog_, sumlogc_;
  };

  class BetaModel
    : public ParamPolicy_2<UnivParams, UnivParams>,
      public SufstatDataPolicy<DoubleData,BetaSuf>,
      public PriorPolicy,
      public NumOptModel,
      public DiffDoubleModel
  {
  public:
    // Initialize with the prior number of "successes" (a) and
    // "failures" (b).  This is the usual parameterization of the Beta
    // model.
    BetaModel(double a = 1.0, double b = 1.0);

    // Initialize the Beta model with a mean and a sample size.  In
    // the standard parameterization, the mean maps to a/(a+b) and the
    // sample size is (a+b).
    BetaModel(double mean, double sample_size, int);

    BetaModel(const BetaModel &m);

    BetaModel *clone() const;

    Ptr<UnivParams> Alpha();
    Ptr<UnivParams> Beta();
    const Ptr<UnivParams> Alpha()const;
    const Ptr<UnivParams> Beta()const;

    const double & a() const;
    const double & b() const;
    void set_a(double alpha);
    void set_b(double beta);
    void set_params(double a, double b);

    // An alternative parameterization:
    //        mean = a/(a+b)
    // sample_size = (a+b)
    double mean()const;         // a/(a+b)
    double sample_size()const;  // a+b
    void set_sample_size(double a_plus_b);
    void set_mean(double a_over_a_plus_b);

    // probability calculations
    double Loglike(Vec &, Mat &, uint) const ;
    double log_likelihood(double a, double b)const;
    double Logp(double x, double &d1, double &d2, uint nd) const ;
    double sim() const;
  private:
    double Logp_degenerate(double x, double &g, double &h, uint nd)const;
  };

  double beta_log_likelihood(double a, double b, const BetaSuf &);
}

#endif// BOOM_BETA_MODEL_HPP
