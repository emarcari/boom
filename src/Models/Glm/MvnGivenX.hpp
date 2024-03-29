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

  //----------------------------------------------------------------------
  // For multinomial logit models there are separate X's for subject
  // characteristics and choice characteristics.  The intercept term
  // is always considered a subject characeristic.  The prior is
  //
  //           beta ~ N(b0, B / prior_sample_size),
  //
  // where b0 and prior_sample_size are specified, and
  //
  // B^{-1} = (1-diagonal_weight) * U + diagonal_weight * diag(U).
  //
  // The matrix U is a block diagonal matrix corresponding roughly to
  // 'unit information'.  There is one block for each choice level
  // (other than choice level 0) corresponding to characteristic of
  // the subject making the choice.  These blocks are identical and
  // equal to X'X/(number of observations * number of choices), where
  // X is the matrix of subject characteristics (or subject
  // covariates).
  //
  // There is an additional block (in the lower right corner) equal to
  //
  // sum_i sum_m w_{im} w_{im}' / (number of observations * number of choices)
  //
  // where w_{im} is the vector of predictors describing choice m
  // faced by subject i.
  class MvnGivenXMultinomialLogit
      : public MvnBase,
        public ParamPolicy_2<VectorParams, UnivParams>,
        public IID_DataPolicy<GlmCoefs>,
        public PriorPolicy
  {
  public:
    MvnGivenXMultinomialLogit(const Vec & beta_prior_mean,
                              double prior_sample_size,
                              double diagonal_weight=0);
    MvnGivenXMultinomialLogit(Ptr<VectorParams> beta_prior_mean,
                              Ptr<UnivParams> prior_sample_size,
                              double diagonal_weight=0);
    MvnGivenXMultinomialLogit(const MvnGivenXMultinomialLogit &rhs);
    virtual MvnGivenXMultinomialLogit * clone()const;

    // Args:
    //   subject_characeristics: An n x p array with rows
    //     corresponding to subjects, and columns to measurements of
    //     each subject.
    //   choice_characteristics: a vector of [nchoices x p] matrices
    //     containing characteristics of the objects to be chosen.
    //   number_of_choices: The number of choices available for the
    //     response.  If choice_characteristics is provided, this
    //     argument must match.
    void set_x(const Mat & subject_characeristics,
               const std::vector<Mat> & choice_characteristics,
               int number_of_choices);

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm()const;
    void set_mu(const Vec &mu);

    Ptr<UnivParams> Kappa_prm();
    const Ptr<UnivParams> Kappa_prm()const;
    double kappa()const;
    void set_kappa(double kappa);

    virtual const Vector & mu()const;
    virtual const Spd & Sigma()const;
    virtual const Spd & siginv()const;
    virtual double ldsi()const;

  private:
    double diagonal_weight_;

    Spd scaled_subject_xtx_;
    Spd scaled_choice_xtx_;

    mutable Spd overall_xtx_;
    mutable bool current_;
    mutable Ptr<SpdData> Sigma_storage_;

    void make_current()const;
  };

}

#endif// BOOM_MVN_GIVEN_X_HPP
