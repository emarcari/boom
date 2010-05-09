/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#ifndef BOOM_BINOMIAL_MIXTURE_SAMPLER_HPP_
#define BOOM_BINOMIAL_MIXTURE_SAMPLER_HPP_

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/BinomialLogitModel.hpp>
#include <Models/Glm/WeightedRegressionModel.hpp>
#include <Models/MvnBase.hpp>

#include <cpputil/math_utils.hpp>

#include <stats/logit.hpp> // for lope

#include <vector>

namespace BOOM{

  struct BinomialLogitComputations{
      BinomialLogitComputations()
          : p0_(10), p1_(10), m0_(10), m1_(10), v0_(10), v1_(10) {}
      double eta_;
      Vec p0_, p1_;
      Vec m0_, m1_;
      Vec v0_, v1_;
      bool operator<(const BinomialLogitComputations &rhs)const{
        return eta_ < rhs.eta_; }
      bool operator<(double eta)const{return eta_ < eta; }
    };



  // draws from the posterior distribution of beta given data in a
  // BinomialLogitModel using the Binomial Auxiliary Mixture Sampling
  // algorithm.
  class BinomialLogitSampler
      : public PosteriorSampler{
   public:
    BinomialLogitSampler(Ptr<BinomialLogitModel> m,
                         Ptr<MvnBase> pri,
                         int clt_threshold = 0); // default is to always batch_impute
    virtual void draw();
    virtual double logpri()const;

    static double mu(int i){return mu_[i];}
    static double sigma(int i){return sigma_[i];}
    static double sigsq(int i){return sigsq_[i];}
    static const Vec & logpi(){return logpi_;}

    static const Vec &mu(){return mu_;}
    static const Vec &sigma(){return sigma_;}
    static const Vec &pi(){return pi_;}
    static const Vec &sigsq(){return sigsq_;}

    void merge_table(const std::vector<BinomialLogitComputations> &);
    double smallest_eta()const;
    double largest_eta()const;
   private:
    void impute_latent_data();
    void draw_beta();

    // impute latent data using large sample approximations to
    // auxiliary mixture sampling.
    void batch_impute(int n, int y, double eta, const Vec &x);
    int locate_position_in_table(double eta);
    void fill_conditional_probs(double eta);
    void compute_conditional_probs(double eta);
    void precompute(double lo, double hi);

    // impute latent data by making trial-level draws using the holmes
    // and held augmentation scheme
    void single_impute(int n, double eta, bool y, const Vec &x);
    double draw_z(bool y, double eta)const;
    double draw_lambda(double r)const;

    void single_impute_auxmix(int n, double eta, bool y, const Vec &x);
    int unmix(double u, Vec &prob)const;

    Ptr<BinomialLogitModel> m_;
    Ptr<MvnBase> pri_;

    std::vector<BinomialLogitComputations> table_;

    // complete data sufficient statistics;
    Spd ivar_, xtwx_;
    Vec ivar_mu_, xtwu_;

    Vec p0_;  // conditional distribution of the mixture
    Vec p1_;   // indicator given eta given eta and y
    Vec cmeans_zero_;
    Vec cmeans_one_;            // conditional means and variances of
    Vec cvars_zero_;            // the latent utility given eta and
    Vec cvars_one_;             // the mixture indicator

    int clt_threshold_;
    double stepsize_;

    // parameters of the Gaussian mixture approximation to the Gumbel
    // distribution
    static const Vec mu_;
    static const Vec sigsq_;
    static const Vec sigma_;
    static const Vec pi_;
    static const Vec logpi_;

    //    static const std::vector<BinomialLogitComputations> initial_table_;
    static std::vector<BinomialLogitComputations> fill_initial_table();
  };

//======================================================================
// helper classes that need to be exposed for testing

  //----------------------------------------------------------------------
  // functor class giving the posterior distribution of the latent
  // utility u given y = 1
  namespace BlsHelper{
    typedef BinomialLogitSampler BLS;
    class u_given_one{
     public:
      u_given_one(double eta) : psi_(lope(eta)) {}

      double prob(double u)const{
        double g,h;
        return exp(logprob(u,g,h,0));
      }

      //TODO(stevescott):  verify derivatives

      double logprob(double u, double &d1, double &d2, int nd)const{
        u -= psi_;
        double eu = exp(-u);
        double ans = -u - eu;
        if(nd>0){
          d1 = 1 - eu;
          if(nd>1) d2 = -eu;
        }
        return ans;
      }

     private:
      double psi_;  // log(1 + exp(eta))
    };

  //----------------------------------------------------------------------
  // functor class giving the posterior distribution of the latent
  // utility u given y = 0
    class u_given_zero{
     public:
      u_given_zero(double eta)
          : eta_(eta), lam_(exp(eta)), psi_(lope(eta)) {}

      double prob(double u)const{
        double eps = exp(-u);
        double log_ans = (eta_ - u) + lope(eta_) - lam_*eps + log(1-exp(-eps));
        return exp(log_ans);
      }

      double logprob(double u, double &g, double &h, int nd)const{
        double eps = exp(-u);
        double E = exp(-eps);
        double emu = eta_-u;
        double expemu = exp(emu);
        double ans = emu + psi_ - expemu + log(1-E);
        if(nd>0){
          g= 1 + exp(eta_ - u) - E * eps/(1-E);
          if(nd>1){
            h= -expemu - E * eps * (E + eps -1)/pow(1-E, 2);
          }
        }
        return ans;
      }

     private:
      double eta_, lam_, psi_;
    };
    //----------------------------------------------------------------------
    // pdf, log pdf, and derivatives of log pdf for the Gaussian mixture
    // approximation to the Gumbel (extreme-value-1) distribution
    class GaussianMixtureLogProb{
     public:
      GaussianMixtureLogProb(int m)
          : m_(m), phi_(10), dphi_(10), d2phi_(10) {}

      double logprob(double u, double &g, double &h, int nd){
        const Vec &pi(BLS::pi());
        //        double max_log_phi = fill_phi(u,nd);
        fill_phi(u,nd);
        double piphi = pi.dot(phi_);
        double ans = log(pi[m_]) + log(phi_[m_]) - log(piphi);
        // max_log_phi_ is not needed because both numerator and
        // denominator are scaled the same way
        const Vec & sigsq(BLS::sigsq());
        if(nd>0){
          double pidphi = pi.dot(dphi_);
          g = -u/sigsq[m_] - pidphi/piphi;

          if(nd>1){
            double pid2phi = pi.dot(d2phi_);
            h = -1/sigsq[m_]  - ( piphi*pid2phi - pidphi * pidphi)/(pow(piphi,2));
          }
        }
        return ans;
      }

      double prob(double u){
        double g,h;
        return exp(logprob(u,g,h,0));
      }

      // fills phi_, dphi_, and d2phi_ with the normal density and its
      // derivatives

      double fill_phi(double u, int nd){
        double max_log_phi = BOOM::infinity(-1);
        const Vec &mu(BLS::mu());
        const Vec &sigma(BLS::sigma());
        const Vec &sigsq(BLS::sigsq());
        for(int m = 0; m < 10; ++m){
          phi_[m] = dnorm(u, mu[m], sigma[m], true);
          max_log_phi = std::max<double>(max_log_phi, phi_[m]);
        }
        for(int i = 0; i < 10; ++i){
          phi_[i] = exp(phi_[i] - max_log_phi);
          if(nd>0){
            dphi_[i] = -(u-mu[i]) * phi_[i]/sigsq[i];
            if(nd>1){
              d2phi_[i] = (phi_[i]/sigsq[i]) * (pow(u-mu[i],2)/sigsq[i] - 1);
            }
          }
        }

        return max_log_phi;
      }
      const Vec &phi()const{return phi_;}
      const Vec &dphi()const{return dphi_;}
      const Vec &d2phi()const{return d2phi_;}
     private:
      int m_;
      Vec phi_;
      Vec dphi_;
      Vec d2phi_;
    };

    //----------------------------------------------------------------------
    class ComponentPosteriorLogProb : public d2ScalarTargetFun{
     public:
      ComponentPosteriorLogProb(double eta, bool y, int m)
          : eta_(eta), y_(y), m_(m)
      {}

      double logp(double u, double &g, double &h, int nd)const{
        GaussianMixtureLogProb q(m_);
        double ans = q.logprob(u-eta_,g,h,nd);
        double d1(0),d2(0);
        if(y_){
          u_given_one logp(eta_);
          ans += logp.logprob(u,d1,d2,nd);
        }else{
          u_given_zero logp(eta_);
          ans += logp.logprob(u,d1,d2,nd);
        }
        if(nd>0){
          g += d1;
          if(nd>1) h += d2;
        }
        return ans;
      }

      double operator()(double u)const{
        double g(0);
        double h(0);
        return logp(u,g,h,0);
      }
      double operator()(double u, double &g)const{
        double h(0);
        return logp(u,g,h,1);
      }
      double operator()(double u, double &g, double &h)const{
        return logp(u,g,h,2);
      }

      double prob(double u)const{
        double g,h;
        return exp(logp(u,g,h,0));
      }

      double first_moment(double u)const{
        return u * prob(u);
      }

      double second_moment(double u, double mu)const{
        return pow(u - mu, 2) * prob(u);
      }
     private:
      double eta_;
      bool y_;
      int m_;
    };

  // this free function does the work in compute_conditional_probs
    void fill_probs(double eta, int i, bool y, Vec &probs, Vec &means, Vec &vars);

  // this function returns a bunch of precomputed results
    std::vector<BinomialLogitComputations> fill_binomial_logit_table();
  } // namespace BlsHelper;

}
#endif // BOOM_BINOMIAL_MIXTURE_SAMPLER_HPP_
