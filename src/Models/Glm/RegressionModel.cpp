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

#include "RegressionModel.hpp"
#include <cpputil/DataTable.hpp>
#include <cpputil/DesignMatrix.hpp>
#include <LinAlg/Types.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp>
#include <Models/GammaModel.hpp>
#include <Models/Glm/MvnGivenXandSigma.hpp>
#include <sstream>
#include <cmath>
#include <distributions.hpp>

namespace BOOM{
  inline void incompatible_X_and_y(const Mat &X, const Vec &y){
    ostringstream out;
    out << "incompatible X and Y" << endl
	<< "X = " << endl << X << endl
	<< "Y = " << endl << y << endl;
    throw std::runtime_error(out.str());
  };

  inline void index_out_of_bounds(uint i, uint bound){
    ostringstream out;
    out << "requested index " << i << " out of bounds." << endl
	<< "bound is " << bound << "."<< endl;
    throw std::runtime_error(out.str());
  };



  Mat add_intercept(const Mat &X){
    Vec one(X.nrow(), 1.0);
    return cbind(one, X); }

  Vec add_intercept(const Vec &x){ return concat(1.0, x); }

  //============================================================
//   RegressionData::RegressionData(double Y, const Vec &X):
//     GlmData(X),
//     y_(Y)
//   {}

//   RegressionData::RegressionData(const RegressionData &rhs)
//     : Data(rhs),
//       GlmData(rhs),
//       y_(rhs.y_)
//   {}

//   RegressionData * RegressionData::clone()const{
//     return new RegressionData(*this);}

//   ostream & RegressionData::display(ostream &out)const{
//     out << y() << " " << x() << " ";
//     return out;}

//   istream & RegressionData::read(istream &in){
//     in >> y_;
//     GlmData::read(in);
//     return in;}

//   RegressionData & RegressionData::operator=(const RegressionData &rhs){
//     if(&rhs==this) return *this;
//     y_ = rhs.y_;
//     GlmData::operator=(rhs);
//     return *this;
//   }

  //============================================================
  anova_table RegSuf::anova()const{
    anova_table ans;
    double nobs = n();
    double p = size();  // p+1 really

    ans.SSE = SSE();
    ans.SST = SST();
    ans.SSM = ans.SST - ans.SSE;
    ans.dft = nobs-1;
    ans.dfe = nobs - p;
    ans.dfm = p-1;
    ans.MSE = ans.SSE/ ans.dfe;
    ans.MSM = ans.SSM/ans.dfm;
    ans.F = ans.MSM/ans.MSE;
    ans.p_value = pf(ans.F, ans.dfm, ans.dfe, false, false);
    return ans;
  }

  ostream & anova_table::display(ostream &out)const{
    out << "ANOVA Table:" << endl
 	<< "\tdf\tSum Sq.\t\tMean Sq.\tF:  " << F << endl
 	<< "Model\t" <<dfm << "\t" <<SSM <<"\t\t" <<MSM<<  endl
 	<< "Error\t" << dfe << "\t"<< SSE <<"\t\t" <<MSE<<"\t p-value: "
 	<< p_value << endl
 	<< "Total\t" << dft <<"\t"<<SST <<endl;
    return out;
  }
  ostream & operator<<(ostream &out, const anova_table &tab){
    tab.display(out);
    return out;}
  //======================================================================


  QrRegSuf::QrRegSuf(const Mat&X, const Vec &y, bool add_icpt):
    qr(add_icpt ? add_intercept(X) : X),
    Qty(),
    sumsqy(0.0),
    current(true)
  {
    Mat Q(qr.getQ());
    Qty = y*Q;
    sumsqy = y.dot(y);
  }

  QrRegSuf::QrRegSuf(const QrRegSuf &rhs)
    : Sufstat(rhs),
      RegSuf(rhs),
      SufstatDetails<DataType>(rhs),
      qr(rhs.qr),
      Qty(rhs.Qty),
      sumsqy(rhs.sumsqy),
      current(rhs.current)
  {}



  uint QrRegSuf::size()const{   // dimension of beta
    //    if(!current) refresh_qr();
    return Qty.size();}

  Spd QrRegSuf::xtx()const{
    //    if(!current) refresh_qr();
    return RTR(qr.getR());}

  Vec QrRegSuf::xty()const{
    //    if(!current) refresh_qr();
    return Qty*qr.getR(); }

  Spd QrRegSuf::xtx(const Selector &inc)const{
    //    if(!current) refresh_qr();
    return RTR(inc.select_square(qr.getR()));
  }

  Vec QrRegSuf::xty(const Selector &inc)const{
    //    if(!current) refresh_qr();
    return inc.select(Qty)*inc.select_square(qr.getR()); }


  double QrRegSuf::yty()const{return sumsqy;}
  void QrRegSuf::clear(){
    sumsqy=0;
    Qty=0;
    qr.clear();
  }

  QrRegSuf * QrRegSuf::clone()const{
    return new QrRegSuf(*this);}


  Vec QrRegSuf::beta_hat()const{
    //if(!current) refresh_qr();
    return qr.Rsolve(Qty); }

  Vec QrRegSuf::beta_hat(const Vec &y)const{
    //    if(!current) refresh_qr();
    return qr.solve(y);}

  void QrRegSuf::Update(const DataType &dp){
    current=false;
    Ptr<DataType> d= dp.clone();
  }  // QR not built for updating

  void QrRegSuf::add_mixture_data(double , const Vec &, double){
    ostringstream err;
    err << "use NeRegSuf for regression model mixture components."
	<< endl;
    throw std::runtime_error(err.str());
  }


  void QrRegSuf::refresh_qr(const std::vector<Ptr<RegressionData> > &raw_data) const {
    if(current) return;
    int n = raw_data.size();  // number of observations
    if(n==0){
      current=false;
      return;}

    Ptr<RegressionData> rdp = DAT(raw_data[0]);
    uint dim_beta = rdp->size();
    Mat X(n, dim_beta);
    Vec y(n);
    sumsqy=0.0;
    for(int i = 0; i<n; ++i){
      rdp = DAT(raw_data[i]);
      y[i] = rdp->y();
      const Vec & x(rdp->x());
      X.set_row(i,x);
//       X(i,0)=1.0;    // this stuff is no longer needed b/c the intercept is explicit
//       int k=0;
//       for(int j=x.lo(); j<=x.hi(); ++j) X(i,++k) = x[j];
      sumsqy += y[i]*y[i];
    }
    qr.decompose(X);
    X = qr.getQ();
    Qty = y*X;
    current=true;
  }

  double QrRegSuf::SSE()const{
    //    if(!current) refresh_qr();
    return sumsqy - Qty.dot(Qty); }

  double QrRegSuf::ybar()const{
    //    if(!current) refresh_qr();
    return qr.getR()(0,0)*Qty[0]/n(); }

  double QrRegSuf::SST()const{
    //    if(!current) refresh_qr();
    return sumsqy - n()*pow(ybar(),2); }

  double QrRegSuf::n()const{
    //    if(!current) refresh_qr();
    return qr.nrow();}

  void QrRegSuf::combine(Ptr<RegSuf>){
    throw std::runtime_error("cannot combine QrRegSuf");
  }

  void QrRegSuf::combine(const RegSuf &){
    throw std::runtime_error("cannot combine QrRegSuf");
  }

  Vec QrRegSuf::vectorize(bool)const{
    throw std::runtime_error("cannot combine QrRegSuf");
    return Vec(1, 0.0);
  }

  Vec::const_iterator QrRegSuf::unvectorize(Vec::const_iterator &v,
                                  bool){
    throw std::runtime_error("cannot combine QrRegSuf");
    return v;
  }

  Vec::const_iterator QrRegSuf::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it(v.begin());
    return unvectorize(it, minimal);
  }


  //---------------------------------------------
  NeRegSuf::NeRegSuf(uint p): xtx_(p), xty_(p), sumsqy(0.0){ }
  NeRegSuf::NeRegSuf(const Mat &X, const Vec &y, bool add_icpt){
    Mat tmpx = add_icpt ? add_intercept(X) : X;
    xty_ =y*tmpx;
    xtx_ = tmpx.inner();
    sumsqy = y.dot(y); }

  NeRegSuf::NeRegSuf(const Spd & XTX, const Vec & XTY, double YTY)
    : xtx_(XTX),
      xty_(XTY),
      sumsqy(YTY)
  {}

  NeRegSuf::NeRegSuf(const NeRegSuf &rhs)
    : Sufstat(rhs),
      RegSuf(rhs),
      SufstatDetails<DataType>(rhs),
      xtx_(rhs.xtx_),
      xty_(rhs.xty_),
      sumsqy(rhs.sumsqy)
  {}

  NeRegSuf * NeRegSuf::clone()const{
    return new NeRegSuf(*this);}


  void NeRegSuf::add_mixture_data(double y, const Vec &x, double prob){
    xtx_.add_outer(x,prob);
    xty_.axpy(x,y*prob);
    sumsqy+= pow(y,2)*prob;
  }

  void NeRegSuf::clear(){xtx_=0.0; xty_=0.0; sumsqy=0.0;}

  void NeRegSuf::Update(const RegressionData &rdp){
    int p = rdp.size();
    if(xtx_.nrow()==0 || xtx_.ncol()==0)
      xtx_ = Spd(p,0.0);
    if(xty_.size()==0) xty_ = Vec(p, 0.0);
    Vec tmpx = rdp.x();// add_intercept(rdp.x());
    double y = rdp.y();
    xty_+= y*tmpx;
    xtx_.add_outer(tmpx);
    sumsqy+= y*y;  }

  uint NeRegSuf::size()const{ return xtx_.ncol();}  // dim(beta)
  Spd NeRegSuf::xtx()const{ return xtx_;}
  Vec NeRegSuf::xty()const{ return xty_;}

  Spd NeRegSuf::xtx(const Selector &inc)const{
    return inc.select(xtx_);}
  Vec NeRegSuf::xty(const Selector &inc)const{
    return inc.select(xty_);}
  double NeRegSuf::yty()const{ return sumsqy;}

  Vec NeRegSuf::beta_hat()const{ return xtx_.solve(xty_); }

  double NeRegSuf::SSE()const{
    Spd ivar = xtx().inv();
    return yty() - ivar.Mdist(xty()); }
  double NeRegSuf::SST()const{ return sumsqy - n()*pow(ybar(),2); }
  double NeRegSuf::n()const{ return xtx_(0,0);  }
  double NeRegSuf::ybar()const{ return xty_[0]/xtx_(0,0);}

  void NeRegSuf::combine(Ptr<RegSuf> sp){
    Ptr<NeRegSuf> s(sp.dcast<NeRegSuf>());
    xtx_ += s->xtx_;
    xty_ += s->xty_;
    sumsqy += s->sumsqy;
  }

  void NeRegSuf::combine(const RegSuf & sp){
    const NeRegSuf& s(dynamic_cast<const NeRegSuf &>(sp));
    xtx_ += s.xtx_;
    xty_ += s.xty_;
    sumsqy += s.sumsqy;
  }

  Vec NeRegSuf::vectorize(bool minimal)const{
    Vec ans = xtx_.vectorize(minimal);
    ans.concat(xty_);
    ans.push_back(sumsqy);
    return ans;
  }

  Vec::const_iterator NeRegSuf::unvectorize(Vec::const_iterator &v,
                                  bool minimal){
    xtx_.unvectorize(v, minimal);
    uint dim = xty_.size();
    xty_.assign(v, v+dim);
    v+=dim;
    sumsqy = *v;  ++v;
    return v;
  }

  Vec::const_iterator NeRegSuf::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }


  //======================================================================
  typedef RegressionDataPolicy RDP;


  RDP::RegressionDataPolicy(Ptr<RegSuf> s)
    : DPBase(s)
  {}
  RDP::RegressionDataPolicy(Ptr<RegSuf>s, const DatasetType &d)
    : DPBase(s,d)
  {}

  RDP::RegressionDataPolicy(const RegressionDataPolicy &rhs)
    : Model(rhs),
      DPBase(rhs)
  {}


  RegressionDataPolicy & RDP::operator=(const RegressionDataPolicy &rhs){
    if(&rhs!=this) DPBase::operator=(rhs);
    return *this;
  }


  //======================================================================
  typedef RegressionModel RM;

  RM::RegressionModel(uint p)
    : GlmModel(),
      ParamPolicy(new GlmCoefs(p), new UnivParams(1.0)),
      DataPolicy(new NeRegSuf(p)),
      ConjPriorPolicy()
  {
  }

  RM::RegressionModel(const Vec &b, double Sigma)
    : GlmModel(),
      ParamPolicy(new GlmCoefs(b), new UnivParams(Sigma*Sigma)),
      DataPolicy(new NeRegSuf(b.size())),
      ConjPriorPolicy()
  {
  }


  RM::RegressionModel(const Mat &X, const Vec &y, bool add_icpt)
    : GlmModel(),
      ParamPolicy(new GlmCoefs(X.ncol()), new UnivParams(1.0)),
      DataPolicy(new QrRegSuf(X,y, add_icpt)),
      ConjPriorPolicy()
  {
    mle();
  }



  RM::RegressionModel(const DesignMatrix &X, const Vec &y, bool add_icpt)
    : GlmModel(),
      ParamPolicy(new GlmCoefs(X.ncol()), new UnivParams(1.0)),
      DataPolicy(new QrRegSuf(X,y, add_icpt)),
      ConjPriorPolicy()
  {
  }


  RM::RegressionModel(const DatasetType &d, bool all)
    : GlmModel(),
      ParamPolicy(new GlmCoefs(d[0]->size(), all), new UnivParams(1.0)),
      DataPolicy(new NeRegSuf(d.begin(), d.end())),
      ConjPriorPolicy()
  {}


  RM::RegressionModel(const RegressionModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      GlmModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      ConjPriorPolicy(rhs),
      NumOptModel(rhs),
      EmMixtureComponent(rhs)
  {
  }

  RM * RM::clone()const{return new RegressionModel(*this); }

  uint RM::nvars()const{ return coef()->nvars(); }
  uint RM::nvars_possible()const{ return coef()->nvars_possible(); }

  Spd RM::xtx(const Selector &inc)const{ return suf()->xtx(inc);}
  Vec RM::xty(const Selector &inc)const{ return suf()->xty(inc);}

  Spd RM::xtx()const{ return xtx( coef()->inc() ) ;}
  Vec RM::xty()const{ return xty( coef()->inc() ) ;}
  double RM::yty()const{ return suf()->yty();  }


  Vec RM::simulate_fake_x()const{
    uint p = nvars_possible();
    Vec x(p-1);
    for(uint i=0; i<p-1; ++i) x[i] = rnorm();
    return x;
  }

  RegressionData * RM::simdat()const{
    Vec x = simulate_fake_x();
    double yhat = predict(x);
    double y = rnorm(yhat, sigma());
    return new RegressionData(y,x);
  }

  RegressionData * RM::simdat(const Vec &X)const{
    double yhat = predict(X);
    double y = rnorm(yhat, sigma());
    return new RegressionData(y,X);
  }


  //======================================================================

  Ptr<GlmCoefs> RM::coef(){return ParamPolicy::prm1();}
  const Ptr<GlmCoefs> RM::coef()const{return ParamPolicy::prm1();}
  void RM::set_sigsq(double s2){ Sigsq_prm()->set(s2);}


  Ptr<UnivParams> RM::Sigsq_prm(){return ParamPolicy::prm2();}
  const Ptr<UnivParams> RM::Sigsq_prm()const {return ParamPolicy::prm2();}

  const double & RM::sigsq()const{return Sigsq_prm()->value();}
  double RM::sigma()const{return sqrt(sigsq());}

  void RM::make_X_y(Mat &X, Vec &Y)const{
    uint p = beta().size();
    uint n = dat().size();
    X = Mat(n,p);
    Y = Vec(n);
    for(uint i=0; i<n; ++i){
      Ptr<RegressionData> rdp = dat()[i];
      const Vec &x(rdp->x());
      assert(x.size()==p);
      X.set_row(i,x);
      Y[i] = rdp->y();
    }
  }

//   RegSuf * RM::create_suf()const{
//     if(strat==QR){
//       std::pair<Mat,Vec> xy(make_X_y());
//       return new QrRegSuf(xy.first,xy.second);
//     }
//     else if(strat==normal_equations){
//       uint p = beta().size();
//       return new NeRegSuf(p);
//     }
//     else return 0;
//   }

//   RegSuf * RM::create_suf(const Mat &X, const Vec &y)const{
//     if(strat==QR) return new QrRegSuf(X,y);
//     else if(strat==normal_equations) return new NeRegSuf(X,y);
//     else return 0;
//   }

  void RM::mle(){
    set_beta(suf()->beta_hat());
    set_sigsq(suf()->SSE()/suf()->n());
  }

  double RM::pdf(dPtr dp, bool logscale)const{
    Ptr<RegressionData> rd = DAT(dp);
    const Vec &x = rd->x();
    return dnorm(rd->y(), predict(x), sigma(), logscale);  }


  double RM::Loglike(Vec &g, Mat &h, uint nd)const{
    const double log2pi = 1.83787706640935;

    const Vec b = this->beta();
    const double sigsq = this->sigsq();
    double n = suf()->n();

    double SSE = yty() - 2*b.dot(xty()) + xtx().Mdist(b);
    double ans =  -.5*(n * log2pi  + n *log(sigsq)+ SSE/sigsq);

    if(nd>0){  // sigsq derivs come first in CP2 vectorization
      Spd xtx = this->xtx();
      Vec gbeta = (xty() - xtx*b)/sigsq;
      double sig4 = sigsq*sigsq;
      double gsigsq = -n/(2*sigsq) + SSE/(2*sig4);
      g = concat(gsigsq, gbeta);
      if(nd>1){
 	double h11 = .5*n/sig4 - SSE/(sig4*sigsq);
 	h = unpartition(h11, (-1/sigsq)*gbeta, (-1/sigsq)*xtx);}}
    return ans;
  }


  void RM::set_conjugate_prior(Ptr<MvnGivenXandSigma> b, Ptr<GammaModel> siginv){
    Ptr<RegressionModel> m(this);
    NEW(RegressionConjSampler, pri)(m, b,siginv);
    this->set_conjugate_prior(pri);
  }

  void RM::set_conjugate_prior(Ptr<RegressionConjSampler> pri){
    ConjPriorPolicy::set_conjugate_prior(pri);
  }

  void RM::add_mixture_data(Ptr<Data> dp, double prob){
    Ptr<RegressionData> d(DAT(dp));
    suf()->add_mixture_data(d->y(), d->x(), prob);
  }

  /*
     SSE = (y-Xb)^t (y-Xb)
     = (y - QQTy)^T (y - Q Q^Ty)
     = ((I=QQ^T)y))^T(I-QQ^T)y)
     = y^T (1-QQ^T)(I-QQ^T)y
     =  y^T ( I - QQ^T - QQ^T +  QQ^TQQ^T)y
     = y^T(I -QQ^T)y

     b = xtx-1xty = (rt qt q r)^{-1} rt qt y
     = r^[-1] qt y
 	
  */

  //======================================================================

} // ends namespace BOOM
