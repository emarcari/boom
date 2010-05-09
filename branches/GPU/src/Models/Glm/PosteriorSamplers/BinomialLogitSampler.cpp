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

  std::vector<BLC> BLS::fill_initial_table(){
    // TODO(stevescott): put this information in the compiler.  Make a
    // table that simply gets read in.
    std::vector<BLC> ans;
    for(double eta = -10; eta <= 10; eta += .1){
      BLC c;
      c.eta_ = eta;
      for(int i = 0; i < 10; ++i){
        fill_probs(eta, i, false, c.p0_, c.m0_, c.v0_);
        fill_probs(eta, i, true, c.p1_, c.m1_, c.v1_);
      }
      c.p0_.normalize_prob();
      c.p1_.normalize_prob();
      ans.push_back(c);
    }
    return ans;
  }

//  const std::vector<BLC> BLS::initial_table_(BLS::fill_initial_table());
//  const std::vector<BLC> BLS::initial_table_(fill_binomial_logit_table());

  BLS::BinomialLogitSampler(Ptr<BinomialLogitModel> m,
                            Ptr<MvnBase> pri,
                            int clt_threshold)
      : m_(m),
        pri_(pri),
        table_(fill_binomial_logit_table()),
        ivar_(m_->xdim()),
        xtwx_(m_->xdim()),
        ivar_mu_(m_->xdim()),
        xtwu_(m_->xdim()),
        p0_(10),
        p1_(10),
        cmeans_zero_(10),
        cmeans_one_(10),
        cvars_zero_(10),
        cvars_one_(10),
        clt_threshold_(clt_threshold),
        stepsize_(.1)
  {
    //    precompute(-10, 10);
  }
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
  int BLS::unmix(double u, Vec &prob)const{
    prob = logpi_;
    for(int i = 0; i < 10; ++i) prob[i] += dnorm(u, mu_[i], sigma_[i], true);
    prob.normalize_logprob();
    return rmulti_mt(rng(), prob);
  }
  //----------------------------------------------------------------------
  void BLS::single_impute_auxmix(int n, double eta, bool y, const Vec &x){
    double sumu = 0;
    double sumn = 0;
    double lam = exp(eta);
    Vec prob(10);
    for(int j = 0; j < n; ++j){
      double emin = rexp(1+lam);
      double u = -log( y ? emin : emin + rexp(lam));
      int z = unmix(u - eta, prob);
      sumu += (u - mu_[z])/sigsq_[z];
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
  void BLS::precompute(double lo, double hi){
    typedef std::vector<BLC>::iterator IT;
    if(hi < lo) std::swap<double>(lo,hi);
    std::vector<BLC> local_table;
    int nsteps = lround(ceil((hi-lo)/stepsize_));
    local_table.reserve(nsteps);

    for(double eta = lo; eta < hi; eta+=stepsize_){
      //      cout << "precomputing eta = " << eta << endl;
      compute_conditional_probs(eta);
      BLC c;
      c.eta_ = eta;
      c.p0_ = p0_;
      c.m0_ = cmeans_zero_;
      c.v0_ = cvars_zero_;
      c.p1_ = p1_;
      c.m1_ = cmeans_one_;
      c.v1_ = cvars_one_;
      local_table.push_back(c);
    }
    merge_table(local_table);
  }
  //----------------------------------------------------------------------
  void BLS::merge_table(const std::vector<BLC> &newtable){
    std::vector<BLC> ans;
    std::set_union(newtable.begin(), newtable.end(),
                   table_.begin(), table_.end(),
                   std::back_inserter(ans));
    table_.swap(ans);
  }

  double BLS::smallest_eta()const{
    if(table_.empty()){
      throw std::runtime_error("you called BinomialLogitSampler::smallest_eta"
                               " without filling the table");
    }
    return table_[0].eta_;
  }

  double BLS::largest_eta()const{
    if(table_.empty()){
      throw std::runtime_error("you called BinomialLogitSampler::largest_eta"
                               " without filling the table");
    }
    return table_.back().eta_;
  }

  //----------------------------------------------------------------------
  // returns an integer 'pos' such that eta is between the tabulated
  // values of pos and pos+1.  If eta is on or outside the boundary of
  // the table then the table is expanded so that eta is in its interior
  int BLS::locate_position_in_table(double eta){
    typedef std::vector<BLC>::iterator IT;
    IT b = table_.begin();
    IT e = table_.end();
    IT it = std::lower_bound(b, e, eta);
    if(it == e){
      double largest_eta = table_.back().eta_;
      precompute(largest_eta + stepsize_, eta + 2*stepsize_);
      // precompute invalidates iterators.  need to reset them
      b = table_.begin();
      it = std::lower_bound(b, table_.end(), eta);
    }else if(it == b){
      // eta is less than the smallest stored value
      double smallest_eta = table_[0].eta_;
      precompute(eta - 2 * stepsize_, smallest_eta - stepsize_);
      // precompute invalidates iterators.  need to reset them
      b = table_.begin();
      it = std::lower_bound(b, table_.end(), eta);
    }
    int pos = it - b - 1;
    return pos;
  }

  //----------------------------------------------------------------------
  // interpolates the conditional probabilities and moments from the
  // relevant tables.  If eta falls outside the range of the relevant
  // tables, the tables are expanded
  void BLS::fill_conditional_probs(double eta){
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
      throw std::runtime_error(err.str());
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
      throw std::runtime_error(err.str());
    }
    p0_ = (1-w) * left.p0_ + w*right.p0_;
    cmeans_zero_ = (1-w) * left.m0_ + w*right.m0_;
    cvars_zero_ = (1-w) * left.v0_ + w*right.v0_;

    p1_ = (1-w) * left.p1_ + w*right.p1_;
    cmeans_one_ = (1-w) * left.m1_ + w*right.m1_;
    cvars_one_ = (1-w) * left.v1_ + w*right.v1_;
  }

  //----------------------------------------------------------------------
  void BLS::batch_impute(int n, int y, double eta, const Vec &x){

    if( (n < 0) || (y > n) ){
      ostringstream err;
      err << "illegal values of n and y in BinomialLogitSampler::batch_impute"
          << endl
          << "n = " << n << endl
          << "y = " << y << endl;
      throw std::runtime_error(err.str());
    }

    fill_conditional_probs(eta);

    std::vector<int> N0(10, 0);
    rmultinom_mt(rng(), n-y, p0_, N0);

    std::vector<int> N1(10, 0);
    rmultinom_mt(rng(), y, p1_, N1);

    double sumu_mean=0;
    double sumu_var=0;
    double sumn = 0;

    for(int m = 0; m < 10; ++m){
      double sig2 = sigsq_[m];
      double sig4 = sig2*sig2;
      double d0 = cmeans_zero_[m] - mu_[m];
      double d1 = cmeans_one_[m] - mu_[m];
      sumu_mean += (N0[m]*d0 + N1[m]*d1)/sig2;
      sumu_var += (N0[m]*cvars_zero_[m] + N1[m]*cvars_one_[m])/sig4;
      sumn += (N0[m] + N1[m])/sig2;
    }

    double sumu = rnorm_mt(rng(), sumu_mean, sqrt(sumu_var));
    xtwu_.axpy(x, sumu);
    xtwx_.add_outer(x, sumn, false);
  }
  //----------------------------------------------------------------------
  void BLS::impute_latent_data(){
    const std::vector<Ptr<BRD> > & data(m_->dat());
    double log_alpha = m_->log_alpha();
    xtwx_=0;
    xtwu_=0;
    Vec beta(m_->beta());
    uint nd = data.size();
    for(uint i = 0; i < nd; ++i){
      Ptr<BRD> dp = data[i];
      const Vec &x(dp->x());
      double eta = m_->predict(x) + log_alpha;
      int y = dp->y();
      int n = dp->n();
      if(n > clt_threshold_){
        batch_impute(n, y, eta, x);
      }else{
//         single_impute(y, eta, 1, x);
//         single_impute(n-y, eta, 0, x);
        single_impute_auxmix(y, eta, 1, x);
        single_impute_auxmix(n-y, eta, 0, x);
      }
    }
    xtwx_.reflect();
  }
  //----------------------------------------------------------------------
  double BLS::draw_z(bool y, double eta)const{
    double trun_prob = plogis(0, eta);
    double u = y ? runif(trun_prob,1) : runif(0,trun_prob);
    return qlogis(u,eta);
  }
  //----------------------------------------------------------------------
  double BLS::draw_lambda(double r)const{
    return Logit::draw_lambda_mt(rng(), r);
  }
  //----------------------------------------------------------------------
  void BLS::draw_beta(){
    ivar_ = pri_->siginv() + xtwx_;
    ivar_mu_ = pri_->siginv()*pri_->mu() + xtwu_;
    ivar_mu_ = rmvn_suf_mt(rng(), ivar_, ivar_mu_);
    m_->set_Beta(ivar_mu_);
  }
  //----------------------------------------------------------------------
  // computes the relevant probabilities and moments needed to do
  // batch imputation
  void BLS::compute_conditional_probs(double eta){
    for(int i = 0; i < 10; ++i){
      fill_probs(eta, i, false, p0_, cmeans_zero_, cvars_zero_);
      fill_probs(eta, i, true, p1_, cmeans_one_, cvars_one_);
    }
//     cout << "eta = " << eta << endl
//          << "p0 = " << p0_ << endl
//          << "p1 = " << p1_ << endl;
    p0_.normalize_prob();
    p1_.normalize_prob();
  }
  //----------------------------------------------------------------------
  // uses quadrature to compute the relevant numerical conditional
  // probabilites and moments for the imputation tables.  This is a
  // driver function for compute_conditional_probs
  namespace BlsHelper{
    void fill_probs(double eta, int i, bool y, Vec &probs, Vec &means, Vec &vars){
      typedef boost::function<double(double)> Fun;
      ComponentPosteriorLogProb f(eta, y, i);

      if(probs.size()!=10) probs.resize(10);
      if(means.size()!=10) means.resize(10);
      if(vars.size()!=10) vars.resize(10);

      Fun p = boost::bind(&ComponentPosteriorLogProb::prob, &f, _1);
      Integral prob0(p, BOOM::infinity(-1), eta);
      Integral prob1(p, eta, BOOM::infinity(1));
      probs[i] = prob0.integrate() + prob1.integrate();

      // settting default values for mu and sigsq in the probs[i] == 0
      // case should be harmless because then N[i] will be zero
      double mu = BLS::mu(i);
      if(probs[i] > 0){
        Fun m = boost::bind(&ComponentPosteriorLogProb::first_moment, &f, _1);
        Integral mu0_integral(m, BOOM::infinity(-1), eta);
        Integral mu1_integral(m, eta, BOOM::infinity(1));
        mu = (mu0_integral.integrate() + mu1_integral.integrate()) / probs[i];
      }
      means[i] = mu;

      double var = BLS::sigsq(i);
      if(probs[i] > 0){
        Fun v = boost::bind(&ComponentPosteriorLogProb::second_moment, &f, _1, mu);
        Integral v0_integral(v, BOOM::infinity(-1), eta);
        Integral v1_integral(v, eta, BOOM::infinity(1));
        var = (v0_integral.integrate() + v1_integral.integrate())/ probs[i];
      }
      vars[i] = var;
    }

  }
}
