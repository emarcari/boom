/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_BETA_BINOMIAL_MODEL_HPP
#define BOOM_BETA_BINOMIAL_MODEL_HPP

#include <Models/DataTypes.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{

  class BinomialData : public Data{
   public:
    BinomialData(int n = 0, int y = 0);
    BinomialData(const BinomialData &rhs);
    BinomialData * clone()const;
    BinomialData & operator=(const BinomialData &rhs);

    virtual uint size(bool minimal = true)const;
    virtual ostream &display(ostream &)const;

    int trials()const;
    int n()const;
    void set_n(int trials);

    int y()const;
    int successes()const;
    void set_y(int successes);
   private:
    int trials_;
    int successes_;

    void check_size(int n, int y)const;
  };

  // BetaBinomialModel describes a setting were binomial data occurs
  // within groups.  Each group has its own binomial success
  // probability drawn from a beta(a, b) distribution.  If the group
  // size is 1 then this is simply the BetaBinomial distribution.

  class BetaBinomialModel
    : public ParamPolicy_2<UnivParams, UnivParams>,
      public IID_DataPolicy<BinomialData>,
      public PriorPolicy,
      public LoglikeModel
  {
   public:
    BetaBinomialModel(double a, double b);
    BetaBinomialModel(const std::vector<int> &trials,
                      const std::vector<int> &successes);
    BetaBinomialModel(const BetaBinomialModel &rhs);
    BetaBinomialModel *clone()const;

    // The likelihood contribution for observation i is
    // int Pr(y_i | theta_i, n_i) p(theta_i) dtheta_i
    virtual double loglike()const;
    double loglike(double a, double b)const;
    double logp(int n, int y, double a, double b)const;

    Ptr<UnivParams> SuccessPrm();
    Ptr<UnivParams> FailurePrm();
    const Ptr<UnivParams> SuccessPrm()const;
    const Ptr<UnivParams> FailurePrm()const;
    double a()const;
    void set_a(double a);
    double b()const;
    void set_b(double b);

    double prior_mean()const;             // a / a+b
    void set_prior_mean(double prob);

    double prior_sample_size()const;      // a+b
    void set_prior_sample_size(double sample_size);

   private:
    void check_positive(double arg, const char *function_name)const;
    void check_probability(double arg, const char *function_name)const;
  };


}

#endif //  BOOM_BETA_BINOMIAL_MODEL_HPP
