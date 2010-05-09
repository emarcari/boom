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
#ifndef BOOM_MVN_GIVEN_X_HPP
#define BOOM_MVN_GIVEN_X_HPP

#include <Models/MvnBase.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{

  class MLogitBase;
  class GlmModel;
  class GlmCoefs;
  class RegressionModel;
  class LogisticRegressionModel;
  class ProbitRegressionModel;

  // This model is intended to serve as a prior for probit and logit
  // regression models p(y | beta, X).  For a prior on Gaussian
  // regression models look at MvnGivenXAndSigma
  //
  // The model is
  // beta ~ N(mu, Ivar)
  // Ivar = Lambda + kappa * [(1-diag_wgt) * xtx
  //                               + diag_wgt*diag(xtx)]) / sumw
  // sumw = sum of w's from add_x
  // Lambda is a diagonal matrix that defaults to 0
  // Ivar = Lambda if sumw==0
  //
  // The justification for this prior is that xtwx is the information
  // in the full complete-data likelihood, so xtwx / sumw is the
  // average information available from a single observation.  If xtwx
  // is highly collinear then it may not be positive definite, so this
  // class offers the option to shrink away from xtwx and towards
  // diag(xtwx).  The amount of shrinkage towards the diagonal.  The
  // Lambda matrix is there to handle cases where no X's have been
  // observed.
  //
  // The "kappa" parameter is a prior sample size.
  // The "Mu" parameter is the prior mean.  Mu decreases in
  // relevance as kappa->0
  //
  // Users of this class must manage xtwx_ explicitly.  For problems
  // where X and weights are fixed then add_x should be called once
  // per row of X shortly after initialization.  For problems where X
  // or w change frequently then clear_xtwx() and add_x() should be
  // called as needed to manage the changes.

  class MvnGivenX
    : public MvnBase,
      public ParamPolicy_2<VectorParams, UnivParams>,
      public IID_DataPolicy<GlmCoefs>,
      public PriorPolicy
  {
  public:
    MvnGivenX(const Vec &Mu, double kappa, double diag_wgt=0);

    MvnGivenX(Ptr<VectorParams> Mu,
	      Ptr<UnivParams> kappa,
	      double diag_wgt=0);
    MvnGivenX(Ptr<VectorParams> Mu,
	      Ptr<UnivParams> kappa,
	      const Vec & Lambda,
	      double diag_wgt=0);
    MvnGivenX(const MvnGivenX &rhs);

    virtual MvnGivenX * clone()const;
    //    virtual void refresh_xtwx()=0;
    virtual void initialize_params();
    virtual void add_x(const Vec &x, double w=1.0);
    virtual void clear_xtwx();
    virtual const Spd & xtwx()const;

    uint dim()const;
    virtual const Vec & mu()const;
    double kappa()const;
    virtual const Spd & Sigma()const;
    virtual const Spd & siginv()const;
    virtual double ldsi()const;

    const Ptr<VectorParams> Mu_prm()const;
    const Ptr<UnivParams> Kappa_prm()const;
    Ptr<VectorParams> Mu_prm();
    Ptr<UnivParams> Kappa_prm();

    double diagonal_weight()const;
    virtual double pdf(Ptr<Data>, bool logscale)const;
    virtual double loglike()const;
    virtual Vec sim()const;
  private:
    virtual void set_ivar()const;  // logical constness

    double diagonal_weight_;
    Vec Lambda_;                // prior if no X's.  may be unallocated

    mutable Ptr<SpdParams> ivar_;
    Spd xtwx_;
    double sumw_;
    mutable bool current_;
  };
  // ------------------------------------------------------------
  // for multinomial logit models
  class MvnGivenXMlogit{
  public:
    MvnGivenXMlogit(MLogitBase *mod,
		     Ptr<VectorParams> beta_prior_mean,
		     Ptr<UnivParams> prior_sample_size,
		     double diag_wgt=0);
    MvnGivenXMlogit(MLogitBase *mod,
		     Ptr<VectorParams> beta_prior_mean,
		     Ptr<UnivParams> prior_sample_size,
		     const Vec & Lambda,
		     double diag_wgt=0);
    MvnGivenXMlogit(const MvnGivenXMlogit &rhs);
    /* virtual */ MvnGivenXMlogit * clone()const;
    /* virtual */ void refresh_xtwx();
  private:
    MLogitBase *ml_;
  };
  //------------------------------------------------------------
  class MvnGivenXLogit
    : public MvnGivenX
  {
  public:
    MvnGivenXLogit(Ptr<LogisticRegressionModel> mod,
		   Ptr<VectorParams> beta_prior_mean,
		   Ptr<UnivParams> prior_sample_size,
		   double diag_wgt=0);

    MvnGivenXLogit(Ptr<LogisticRegressionModel> mod,
		   Ptr<VectorParams> beta_prior_mean,
		   Ptr<UnivParams> prior_sample_size,
		   const Vec & Lambda,
		   double diag_wgt=0);
    MvnGivenXLogit(const MvnGivenXLogit &rhs);

    /* virtual */ MvnGivenXLogit * clone()const;
    /* virtual */ void refresh_xtwx();
  private:
    Ptr<LogisticRegressionModel> mod_;
  };


}

#endif// BOOM_MVN_GIVEN_X_HPP
