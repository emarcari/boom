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
#ifndef BOOM_BINOMIAL_MODEL_HPP
#define BOOM_BINOMIAL_MODEL_HPP

#include "ModelTypes.hpp"
#include "Sufstat.hpp"
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{

  class BinomialSuf : public SufstatDetails<IntData>{
  public:
    BinomialSuf();
    BinomialSuf(const BinomialSuf &rhs);
    BinomialSuf * clone()const;
    double sum()const;
    double nobs()const;
    virtual void clear();
    virtual void Update(const IntData &);
    void update_raw(double y);

    BinomialSuf * abstract_combine(Sufstat *s){
      return abstract_combine_impl(this, s);}
    void combine(Ptr<BinomialSuf>);
    void combine(const BinomialSuf &);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
  private:
    double sum_, nobs_;
  };

  class BinomialModel
    : public ParamPolicy_1<UnivParams>,
      public SufstatDataPolicy<IntData,BinomialSuf>,
      public PriorPolicy,
      public NumOptModel
  {
  public:
    BinomialModel(uint n=1, double p=.5);
    BinomialModel(const BinomialModel &rhs);
    BinomialModel * clone()const;

    virtual void mle();
    virtual double Loglike(Vec &g, Mat &h, uint nd)const;

    uint n()const;
    double prob()const;
    void set_prob(double p);

    virtual double pdf(Ptr<Data> x, bool logscale)const;
    double pdf(uint x, bool logscale)const;

    Ptr<UnivParams> Prob_prm();
    const Ptr<UnivParams> Prob_prm()const;
    uint sim()const;

  private:
    const uint n_;
  };

}

#endif // BOOM_BINOMIAL_MODEL_HPP
