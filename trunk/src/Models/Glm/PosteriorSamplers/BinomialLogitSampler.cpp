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
#include <Models/Glm/PosteriorSamplers/BinomialLogitSampler.hpp>
#include <Models/Glm/PosteriorSamplers/draw_logit_lambda.hpp>
#include <numopt/Integral.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <algorithm>
#include <boost/thread.hpp>

namespace BOOM{

  // using declaration brings helper classes and typedef BLS into scope
  using namespace BlsHelper;
  typedef BinomialRegressionData BRD;
  typedef BinomialLogitComputations BLC;

  const Vec BLS::mu_("5.09 3.29 1.82 1.24 0.76 0.39 "
                     "0.04 -0.31 -0.67  -1.06");
  const Vec BLS::sigsq_(Vec("4.5 2.02 1.1 0.42 0.2 0.11 "
                            "0.08 0.08 0.09 0.15"));
  const Vec BLS::sigma_(pow(BLS::sigsq_,0.5));
  const Vec BLS::pi_("0.004 0.04 0.168 0.147 0.125 0.101 "
                     "0.104 0.116 0.107 0.088");
  const Vec BLS::logpi_(log(BLS::pi_));

  const std::vector<BLC> BLS::table_(fill_binomial_logit_table());

  void BinomialLogitDataImputer::operator()(){
    sampler_->impute_latent_data_driver(worker_number_, number_of_workers_);
  }

  void single_impute_auxmix(int n, double eta, bool y, const Vec &x,
                            Vec &xtwu_, Spd & xtwx_, const Vec &beta);

  BLS::BinomialLogitSampler(BinomialLogitModel *m,
                            Ptr<MvnBase> pri,
                            int clt_threshold)
      : m_(m),
        beta_(m->coef_prm().get()),
        pri_(pri),
        xtwx_(m_->xdim()),
        xtwu_(m_->xdim()),
        clt_threshold_(clt_threshold),
        ivar_(m_->xdim()),
        ivar_mu_(m_->xdim())
  {}
  //----------------------------------------------------------------------
  void BLS::draw(){
    impute_latent_data();
    draw_beta();
  }
  //----------------------------------------------------------------------
  double BLS::logpri()const{
    return pri_->logp(m_->Beta());
  }
  //----------------------------------------------------------------------
  void BLS::set_number_of_threads(int nworkers){
    workers_.clear();
    if(nworkers > 0){
      for (int i = 0; i < nworkers; ++i){
        NEW(BinomialLogitSampler, worker)(m_, pri_, clt_threshold_);
        // A single threaded worker
        workers_.push_back(worker);
        long seed = static_cast<long>(time(NULL)) * (i+1);
        worker->set_seed(seed);
      }
    }
  }
  //----------------------------------------------------------------------
  int BLS::unmix(double u, Vec &prob)const{
    prob = logpi_;
    for(int i = 0; i < 10; ++i) prob[i] += dnorm(u, mu_[i], sigma_[i], true);
    prob.normalize_logprob();
    return rmulti_mt(rng(), prob);
  }
  //----------------------------------------------------------------------
  void BLS::print_data(std::ostream &out)const{
    const std::vector<Ptr<BinomialRegressionData> > &data(m_->dat());
    int n = data.size();
    out << "data:" << endl;
    for(int i = 0; i < n; ++i) out << *(data[i]) << endl;
  }
  //----------------------------------------------------------------------
  void BLS::single_impute_auxmix(int n, double eta, bool y, const Vec &x){
    double sumu = 0;
    double sumn = 0;
    double lam = exp(eta);
    if(!finite(lam) || lam == 0){
      ostringstream msg;
      msg << "overflow in BinomialLogitSampler::single_impute_auxmix" << endl
          << "eta = " << eta << endl
          << "y   = " << y << endl
          << "x   = " << x << endl
          << "beta = " << m_->Beta() << endl;
      print_data(msg);
      report_error(msg.str());
    }
    Vec prob(10);
    for(int j = 0; j < n; ++j){
      double emin = rexp_mt(rng(), 1+lam);
      double u = -log(y ? emin : emin + rexp_mt(rng(), lam));
      int z = unmix(u - eta, prob);
      double du = (u - mu_[z])/sigsq_[z];

      if(u > 10000 || u < -10000) {
        ostringstream msg;
        msg << "got a crazy value of u in "
            << "BinomialLogitSampler::single_impute_auxmix" << endl
            << "u = " << u << endl
            << "du = " << du << endl
            << "z = " << z << endl
            << "mu_[z] = " << mu_[z] << endl
            << "sigsq_[z] = " << sigsq_[z] << endl
            << "prob = " << prob << endl
            << "eta = " << eta << endl
            << "n   = " << n << endl
            << "y   = " << y << endl
            << "x   = " << x << endl
            << "beta = " << m_->Beta() << endl;
        print_data(msg);
        report_error(msg.str());
      }
      sumu += du;
      sumn += 1.0/sigsq_[z];
    }
    xtwu_.axpy(x, sumu);
    xtwx_.add_outer(x, sumn, false);
  }
  //----------------------------------------------------------------------
  void BLS::single_impute(int n, double eta, bool y, const Vec &x){
    double sumz=0;
    double sumw=0;
    for(uint j = 0; j < n; ++j){
      double z = draw_z(y, eta);
      double lam = draw_lambda(fabs(y-eta));
      double w = 1.0/lam;
      sumz+= z*w;
      sumw+= w;
    }
    xtwu_.axpy(x,sumz);
    xtwx_.add_outer(x,sumw, false);
  }
  //----------------------------------------------------------------------
  // returns an integer 'pos' such that eta is between the tabulated
  // values of pos and pos+1.  If eta is on or outside the boundary of
  // the table then the table is expanded so that eta is in its interior
  int BLS::locate_position_in_table(double eta){
    typedef std::vector<BLC>::const_iterator IT;
    IT b = table_.begin();
    IT e = table_.end();
    IT it = std::lower_bound(b, e, eta);
    if(it == e){
      report_error("BLS fell off right end of table");
    }else if(it == b){
      report_error("BLS fell off left end of table");
    }
    int pos = it - b - 1;
    return pos;
  }

  //----------------------------------------------------------------------
  // interpolates the conditional probabilities and moments from the
  // relevant tables.  If eta falls outside the range of the relevant
  // tables, the tables are expanded
  void BLS::fill_conditional_probs(double eta){
    if(eta <= table_.front().eta_){
      left_endpoint(eta);
      return;
    }else if(eta >= table_.back().eta_){
      right_endpoint(eta);
      return;
    }
    int pos = locate_position_in_table(eta);
    const BLC &left(table_[pos]);
    const BLC &right(table_[pos+1]);

    if( (eta < left.eta_) || (eta > right.eta_) || (right.eta_ < left.eta_) ){
      ostringstream err;
      err << "something is wrong in BinomialLogitSampler::"
          << "fill_conditional_probs..." << endl
          << "left  = " << left.eta_ << endl
          << "eta   = " << eta << endl
          << "right = " << right.eta_ << endl
          << "should have left <= eta <= right" << endl;
      report_error(err.str());
    }

    // w is the weight on the right endpoint
    double w = (eta - left.eta_)/(right.eta_-left.eta_);
    if(w < 0 || w > 1){
      std::ostringstream err;
      err << "something is wrong with w in "
          << "BinomialLogitSampler::fill_conditional_probs" << endl
          << "eta = " << eta <<endl
          << "eta should be in (" << left.eta_
          << ", " << right.eta_ << ")" <<endl
          ;
      report_error(err.str());
    }
    b_.p0_ = (1-w) * left.p0_ + w*right.p0_;
    b_.m0_ = (1-w) * left.m0_ + w*right.m0_;
    b_.v0_ = (1-w) * left.v0_ + w*right.v0_;

    b_.p1_ = (1-w) * left.p1_ + w*right.p1_;
    b_.m1_ = (1-w) * left.m1_ + w*right.m1_;
    b_.v1_ = (1-w) * left.v1_ + w*right.v1_;
  }

  void BLS::left_endpoint(double eta){
    b_ = table_.front();
    b_.m1_ = mu_ + eta;
  }

  void BLS::right_endpoint(double eta){
    b_ = table_.back();
    b_.m1_ = mu_ + eta;
    b_.m0_ += (eta - b_.eta_);
  }
  //----------------------------------------------------------------------
  void BLS::batch_impute(int n, int y, double eta, const Vec &x){

    if( (n < 0) || (y > n) ){
      ostringstream err;
      err << "illegal values of n and y in BinomialLogitSampler::batch_impute"
          << endl
          << "n = " << n << endl
          << "y = " << y << endl;
      report_error(err.str());
    }

    fill_conditional_probs(eta);

    std::vector<int> N0(10, 0);
    rmultinom_mt(rng(), n-y, b_.p0_, N0);

    std::vector<int> N1(10, 0);
    rmultinom_mt(rng(), y, b_.p1_, N1);

    double sumu_mean=0;
    double sumu_var=0;
    double sumn = 0;

    for(int m = 0; m < 10; ++m){
      double sig2 = sigsq_[m];
      double sig4 = sig2*sig2;
      double d0 = b_.m0_[m] - mu_[m];
      double d1 = b_.m1_[m] - mu_[m];
      sumu_mean += (N0[m]*d0 + N1[m]*d1)/sig2;
      sumu_var += (N0[m]*b_.v0_[m] + N1[m]*b_.v1_[m])/sig4;
      sumn += (N0[m] + N1[m])/sig2;
    }

    double sumu = rnorm_mt(rng(), sumu_mean, sqrt(sumu_var));
    xtwu_.axpy(x, sumu);
    xtwx_.add_outer(x, sumn, false);
  }
  //----------------------------------------------------------------------
  void BLS::impute_latent_data(){
    if(!workers_.empty()){
      impute_latent_data_distributed();
    } else {
      impute_latent_data_driver(0, 1);
    }
  }
  //----------------------------------------------------------------------
  void BLS::impute_latent_data_distributed(){
    boost::thread_group tg;
    for(uint i = 0; i < workers_.size(); ++i){
      BinomialLogitDataImputer worker(workers_[i].get(), i,
                                      workers_.size());
      tg.add_thread(new boost::thread(worker));
    }
    tg.join_all();
    xtwu_ = 0;
    xtwx_ = 0;
    for(uint i = 0; i < workers_.size(); ++i){
      xtwu_ += workers_[i]->xtwu_;
      xtwx_ += workers_[i]->xtwx_;
    }
  }
  //----------------------------------------------------------------------
  void BLS::impute_latent_data_driver(int worker_id, int number_of_workers){
    const std::vector<Ptr<BRD> > & data(m_->dat());
    double log_alpha = m_->log_alpha();
    xtwx_=0;
    xtwu_=0;
    uint nd = data.size();
    for(uint i = 0; i < nd; ++i){
      if((i % number_of_workers) == worker_id){
        Ptr<BRD> dp = data[i];
        int n = dp->n();
        if(n==0) continue;
        int y = dp->y();
        const Vec &x(dp->x());
        double eta = beta_->predict(x) + log_alpha;
        if(n > clt_threshold_){
          batch_impute(n, y, eta, x);
        }else{
          //  There was a problem with single_impute (holmes and held
          //  algorithm).  It did not return the correct answer on simulated
          //  data.
          single_impute_auxmix(y, eta, 1, x);
          single_impute_auxmix(n-y, eta, 0, x);
        }
      }
    }
    xtwx_.reflect();
  }
  //----------------------------------------------------------------------
  // latent logistic "relative utility"
  double BLS::draw_z(bool y, double eta)const{
    double trun_prob = plogis(0, eta);
    double u = y ? runif_mt(rng(),trun_prob,1) : runif_mt(rng(),0,trun_prob);
    return qlogis(u,eta);
  }
  //----------------------------------------------------------------------
  double BLS::draw_lambda(double r)const{
    return Logit::draw_lambda_mt(rng(), r);
  }
  //----------------------------------------------------------------------
  // conditional draw of model coefficients given complete data
  // sufficient statistics
  void BLS::draw_beta(){
    ivar_ = pri_->siginv() + xtwx_;
    ivar_mu_ = pri_->siginv()*pri_->mu() + xtwu_;
    ivar_mu_ = rmvn_suf_mt(rng(), ivar_, ivar_mu_);
    m_->set_Beta(ivar_mu_);
  }

  const Spd & BLS::xtwx()const{return xtwx_;}
  const Vec & BLS::xtwu()const{return xtwu_;}

  //----------------------------------------------------------------------
  // free function that uses quadrature to compute the relevant
  // numerical conditional probabilites and moments for the imputation
  // tables.

  namespace BlsHelper{
  const double really_small_probability = 1e-100;
  //======================================================================
    BLC fill_probs(double eta, const BLC *last){
      BLC ans;
      ans.eta_ = eta;
      bool have_last = (last != NULL);
      for(int i = 0; i < 10; ++i){
        fill_probs(eta, i, false, ans.p0_, ans.m0_, ans.v0_,
                   have_last ? last->m0_[i] : BOOM::infinity(1));

        fill_probs(eta, i, true, ans.p1_, ans.m1_, ans.v1_,
                   have_last ? last->m1_[i] : BOOM::infinity(1));
        if(last && ans.p1_[i] <= really_small_probability){
          ans.m1_[i] = last->m1_[i];
          ans.v1_[i] = last->v1_[i];
        }
      }

      // normalize the probability distributions to appease a finicky
      // rmultinom
      double nc = sum(ans.p0_);
      if(fabs(nc - 1) > .01){
        std::ostringstream msg;
        msg << "p0 does not sum to 1 in BlsHelper::fill_probs." << endl
            << "for eta = " << eta << endl
            << "p0 = " << ans.p0_ << endl
            << "sum(p0) = " << nc << endl
            ;
        report_error(msg.str());
      }
      ans.p0_ /= nc;
      nc = sum(ans.p1_);
      if(fabs(nc - 1) > .01){
        std::ostringstream msg;
        msg << "p1 does not sum to 1 in BlsHelper::fill_probs." << endl
            << "for eta = " << eta << endl
            << "p1 = " << ans.p1_ << endl
            << "sum(p1) = " << nc << endl
            ;
        report_error(msg.str());
      }
      ans.p1_ /= nc;
      return ans;
    }
  //======================================================================
    void fill_probs(double eta, int i, bool y, Vec &probs,
                    Vec &means, Vec &vars, double last_mu){
      typedef boost::function<double(double)> Fun;
      ComponentPosteriorLogProb f(eta, y, i);

      if(probs.size()!=10) probs.resize(10);
      if(means.size()!=10) means.resize(10);
      if(vars.size()!=10) vars.resize(10);

      if(last_mu == BOOM::infinity(1)) last_mu = eta;

      Fun p = boost::bind(&ComponentPosteriorLogProb::prob, &f, _1);
      Integral prob0(p, BOOM::infinity(-1), last_mu, 1000);
      //      Integral prob0(p, BOOM::infinity(-1), mode);
      prob0.throw_on_error(false);
      Integral prob1(p, last_mu, BOOM::infinity(1), 1000);
      //Integral prob1(p, mode, BOOM::infinity(1));
      prob1.throw_on_error(false);
      probs[i] = prob0.integrate() + prob1.integrate();
      if(prob0.error_code() || prob1.error_code()){
        std::ostringstream msg;
        msg << "BlsHelper::fill_probs: error compuing probabilities" << endl
            << "eta = " << eta << endl
            << "prob0 integral:" << endl
            << prob0.debug_string().c_str() << endl
            << "prob1 integral:" << endl
            << prob1.debug_string().c_str() << endl;
        report_error(msg.str());
      }

      // settting default values for mu and sigsq in the probs[i] == 0
      // case should be harmless because then N[i] will be zero
      double mu = eta + BLS::mu(i);
      if(probs[i] > really_small_probability){
        Fun m = boost::bind(&ComponentPosteriorLogProb::first_moment, &f, _1);
        Integral mu0_integral(m, BOOM::infinity(-1), last_mu, 1000);
        mu0_integral.throw_on_error(false);
        Integral mu1_integral(m, last_mu, BOOM::infinity(1), 1000);
        mu1_integral.throw_on_error(false);
        mu = (mu0_integral.integrate() + mu1_integral.integrate()) / probs[i];
        if(mu0_integral.error_code() || mu1_integral.error_code()){
          std::ostringstream msg;
          msg << "problem computing conditional means in BlsHelper::fill_probs"
              << endl
              << "eta = " << eta << endl
              << "an exception was thrown with the following error message: "
              << endl
              << mu0_integral.error_message() << endl
              << mu1_integral.error_message() << endl
              << "mu0 integral:" << endl << mu0_integral.debug_string() << endl
              << "mu1 integral:" << endl << mu1_integral.debug_string() << endl;
          report_error(msg.str());
        }
      }
      means[i] = mu;

      double var = BLS::sigsq(i);
      if(probs[i] > really_small_probability){
        Fun v = boost::bind(&ComponentPosteriorLogProb::second_moment,
                            &f, _1, mu);
        Integral v0_integral(v, BOOM::infinity(-1), mu, 1000);
        v0_integral.throw_on_error(false);
        Integral v1_integral(v, mu, BOOM::infinity(1), 1000);
        v1_integral.throw_on_error(false);
        var = (v0_integral.integrate() + v1_integral.integrate())/ probs[i];
        if(v0_integral.error_code() || v1_integral.error_code()){
          ostringstream msg;
          msg << "problem computing conditional variances in "
              << "BlsHelper::fill_probs." << endl
              << "eta = " << eta << endl
              << "an exception was thrown with the following error message:"
              << endl
              << v0_integral.error_message() << endl
              << v1_integral.error_message() << endl
              << "v0 integral:" << endl << v0_integral.debug_string() << endl
              << "v1 integral:" << endl << v1_integral.debug_string() << endl;
          report_error(msg.str());
        }
      }
      vars[i] = var;
    }
  }
}
