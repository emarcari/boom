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
#include <Models/Glm/BinomialLogitModel.hpp>
#include <distributions.hpp>
#include <stats/logit.hpp>

namespace BOOM{
  typedef BinomialLogitModel BLM;
  typedef BinomialRegressionData BRD;
  typedef BinomialLogitLogLikelihood BLLL;


  BLLL::BinomialLogitLogLikelihood(const BLM *m)
      : m_(m)
  {}

  double BLLL::operator()(const Vec & beta)const{
    return m_->log_likelihood(beta, 0, 0);
  }
  double BLLL::operator()(const Vec & beta, Vec &g)const{
    return m_->log_likelihood(beta, &g, 0);
  }
  double BLLL::operator()(const Vec & beta, Vec &g, Mat &H)const{
    return m_->log_likelihood(beta, &g, &H);
  }

  BLM::BinomialLogitModel(uint beta_dim, bool all)
    : ParamPolicy(new GlmCoefs(beta_dim, all)),
      log_alpha_(0)
  {}

  BLM::BinomialLogitModel(const Vec &beta)
    : ParamPolicy(new GlmCoefs(beta)),
      log_alpha_(0)
  {}


  BLM::BinomialLogitModel
  (const Mat &X, const Vec &y, const Vec &n, bool add_int)
    : ParamPolicy(new GlmCoefs(X.ncol() + add_int)),
      log_alpha_(0)
  {
    int nr = nrow(X);
    for(int i = 0; i < nr; ++i){
      uint yi = lround(y[i]);
      uint ni = lround(n[i]);
      NEW(BinomialRegressionData, dp)(yi, ni, X.row(i));
      add_data(dp);
    }
  }

  BLM::BinomialLogitModel
  (const BLM &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      GlmModel(rhs),
      NumOptModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      log_alpha_(0)
  {}

  BLM* BinomialLogitModel::clone()const{
    return new BinomialLogitModel(*this);}


  double BLM::pdf(dPtr dp, bool logscale) const{
    return pdf(DAT(dp), logscale);}

  double BLM::pdf(Ptr<BRD> dp, bool logscale) const{
    return logp(dp->y(), dp->n(), dp->x(), logscale); }

  double BLM::logp_1(bool y, const Vec &x, bool logscale)const{
    double btx = predict(x);
    double ans = -lope(btx);
    if(y) ans += btx;
    return logscale ? ans : exp(ans);
  }

  double BLM::logp(uint y, uint n, const Vec &x, bool logscale)const{
    double eta = predict(x);
    double p = logit_inv(eta);
    return dbinom(y, n, p, logscale);
  }

  double BLM::Loglike(Vec &g, Mat &h, uint nd)const{
    if(nd>=2) return log_likelihood(Beta(), &g, &h);
    if(nd==1) return log_likelihood(Beta(), &g, 0);
    return log_likelihood(Beta(), 0, 0);
  }

  double BLM::log_likelihood(const Vec & beta, Vec *g, Mat *h,
                             bool initialize_derivs)const{
    const BLM::DatasetType &data(dat());
    if(initialize_derivs){if(g){ *g=0; if(h){ *h=0;}}}

    double ans = 0;
    for(int i = 0; i < data.size(); ++i){
      // y and n had been defined as uint's but y-n*p was computing
      // -n, which overflowed
      int y = data[i]->y();
      int n = data[i]->n();
      const Vec & x(data[i]->x());
      double eta = predict(x) - log_alpha_;
      double p = logit_inv(eta);
      double loglike = dbinom(y, n, p, true);
      ans += loglike;
      if(g){
        g->axpy(x, y-n*p);  // g += (y-n*p) * x;
        if(h){ h->add_outer(x,x, -n*p*(1-p)); // h += -npq * x x^T
        }}}
    return ans;
  }

  BLLL BLM::log_likelihood_tf()const{ return BLLL(this); }

  Spd BLM::xtx()const{
    const std::vector<Ptr<BinomialRegressionData> > & d(dat());
    uint n = d.size();
    uint p = d[0]->xdim();
    Spd ans(p);
    for(uint i=0; i<n; ++i){
      double n = d[i]->n();
      ans.add_outer(d[i]->x(), n, false);
    }
    ans.reflect();
    return ans;
  }


  void BLM::set_nonevent_sampling_prob(double alpha){
    if(alpha <=0 || alpha > 1){
      ostringstream err;
      err << "alpha (proportion of non-events retained in the data) "
          << "must be in (0,1]" << endl
          << "you set alpha = " << alpha << endl;
      throw std::runtime_error(err.str());
    }
    log_alpha_ = std::log(alpha);
  }

  double BLM::log_alpha()const{return log_alpha_;}


}
