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

#ifndef BOOM_R_PRIOR_SPECIFICATION_HPP_
#define BOOM_R_PRIOR_SPECIFICATION_HPP_

#include <Interfaces/R/boom_r_tools.hpp>

namespace BOOM{

  class MarkovModel;

  namespace RInterface{
    // Convenience classes for communicating commonly used R objects
    // to BOOM.  Each object has a corresponding R function that will
    // create the SEXP that the C++ object can use to build itself.

    // For encoding an inverse Gamma prior on a variance parameter.
    // See the R help file for CreateSdPrior.
    class SdPrior {
     public:
      explicit SdPrior(SEXP sd_prior);
      double prior_guess()const {return prior_guess_;}
      double prior_df()const {return prior_df_;}
      double initial_value()const {return initial_value_;}
      bool fixed()const {return fixed_;}
      double upper_limit()const {return upper_limit_;}
      ostream & print(ostream &out)const;
     private:
      double prior_guess_;
      double prior_df_;
      double initial_value_;
      bool fixed_;
      double upper_limit_;
    };
    //----------------------------------------------------------------------

    // For encoding a Gaussian prior on a scalar.  See the R help file
    // for CreateNormalPrior.
    class NormalPrior {
     public:
      explicit NormalPrior(SEXP prior);
      virtual ~NormalPrior() {}
      virtual ostream & print(ostream &out) const;
      double mu()const {return mu_;}
      double sigma()const {return sigma_;}
      double initial_value()const {return initial_value_;}
     private:
      double mu_;
      double sigma_;
      double initial_value_;
    };
    //----------------------------------------------------------------------
    // For encoding a prior on an AR1 coefficient.  This is a Gaussian
    // prior, but users have the option of truncating the support to
    // [-1, 1] to enforce stationarity of the AR1 process.
    class Ar1CoefficientPrior : public NormalPrior {
     public:
      explicit Ar1CoefficientPrior(SEXP prior);
      bool force_stationary()const {return force_stationary_;}
      bool force_positive()const {return force_positive_;}
      ostream & print(ostream &out)const;
     private:
      bool force_stationary_;
      bool force_positive_;
    };
    //----------------------------------------------------------------------
    // For encoding the parameters in a conditionally normal model.
    // Tyically this is the prior on mu in an normal(mu, sigsq), where
    // mu | sigsq ~ N(mu0, sigsq / sample_size).
    class ConditionalNormalPrior {
     public:
      explicit ConditionalNormalPrior(SEXP prior);
      double prior_mean()const{return mu_;}
      double sample_size()const{return sample_size_;}
      ostream & print(ostream &out)const;
     private:
      double mu_;
      double sample_size_;
    };

    //----------------------------------------------------------------------
    // A NormalInverseGammaPrior is the conjugate prior for the mean
    // and variance in a normal distribution.
    class NormalInverseGammaPrior {
     public:
      explicit NormalInverseGammaPrior(SEXP prior);
      double prior_mean_guess()const{return prior_mean_guess_;}
      double prior_mean_sample_size()const{return prior_mean_sample_size_;}
      const SdPrior &sd_prior()const{return sd_prior_;}
      ostream & print(ostream &out)const;
     private:
      double prior_mean_guess_;
      double prior_mean_sample_size_;
      SdPrior sd_prior_;
    };

    // For encoding the parameters of a Dirichlet distribution.  The R
    // constructor that builds 'prior' ensures that prior_counts_ is a
    // positive length vector of positive reals.
    class DirichletPrior {
     public:
      explicit DirichletPrior(SEXP prior);
      const Vec & prior_counts()const;
      int dim()const;
     private:
      Vec prior_counts_;
    };

    //----------------------------------------------------------------------
    // For encoding a prior on the parameters of a Markov chain.  This
    // is product Dirichlet prior for the rows of the transition
    // probabilities, and an independent Dirichlet on the initial
    // state distribution.
    // TODO(stevescott): add support for fixing the initial
    //   distribution in various ways.
    class MarkovPrior {
     public:
      explicit MarkovPrior(SEXP prior);
      const Mat & transition_counts()const {return transition_counts_;}
      const Vec & initial_state_counts()const {return initial_state_counts_;}
      ostream & print(ostream &out)const;
      // Creates a Markov model with this as a prior.
      BOOM::MarkovModel * create_markov_model()const;
     private:
      Mat transition_counts_;
      Vec initial_state_counts_;
    };

    class BetaPrior {
     public:
      explicit BetaPrior(SEXP prior);
      double a()const{return a_;}
      double b()const{return b_;}
      ostream & print(ostream &out)const;
     private:
      double a_, b_;
    };

    class GammaPrior {
     public:
      explicit GammaPrior(SEXP prior);
      double a()const{return a_;}
      double b()const{return b_;}
      double initial_value()const{return initial_value_;}
      ostream & print(ostream &out)const;
     private:
      double a_, b_;
      double initial_value_;
    };

    class MvnPrior {
     public:
      explicit MvnPrior(SEXP prior);
      const Vec & mu()const{return mu_;}
      const Spd & Sigma()const{return Sigma_;}
      ostream & print(ostream &out)const;
     private:
      Vec mu_;
      Spd Sigma_;
    };

    class NormalInverseWishartPrior {
     public:
      NormalInverseWishartPrior(SEXP prior);
      const Vec & mu_guess()const{return mu_guess_;}
      double mu_guess_weight()const{return mu_guess_weight_;}
      const Spd & Sigma_guess()const{return sigma_guess_;}
      double Sigma_guess_weight()const{return sigma_guess_weight_;}
      ostream & print(ostream &out)const;
     private:
      Vec mu_guess_;
      double mu_guess_weight_;
      Spd sigma_guess_;
      double sigma_guess_weight_;
    };

    inline ostream & operator<<(ostream &out, const NormalPrior &p) {
      return p.print(out); }
    inline ostream & operator<<(ostream &out, const SdPrior &p) {
      return p.print(out); }
    inline ostream & operator<<(ostream &out, const BetaPrior &p) {
      return p.print(out); }
    inline ostream & operator<<(ostream &out, const MarkovPrior &p) {
      return p.print(out); }
    inline ostream & operator<<(ostream &out, const ConditionalNormalPrior &p) {
      return p.print(out); }
    inline ostream & operator<<(ostream &out, const MvnPrior &p) {
      return p.print(out); }
  }
}

#endif // BOOM_R_PRIOR_SPECIFICATION_HPP_
