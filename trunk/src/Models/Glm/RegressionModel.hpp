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

#ifndef REGRESSIION_MODEL_H
#define REGRESSIION_MODEL_H

#include <BOOM.hpp>
#include "Glm.hpp"
#include <LinAlg/QR.hpp>
#include <Models/Sufstat.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/EmMixtureComponent.hpp>

namespace BOOM{

  class RegressionConjSampler;
  class DesignMatrix;
  class MvnGivenXandSigma;
  class GammaModel;

  class anova_table{
   public:
    double SSE, SSM, SST;
    double MSM, MSE;
    double dfe, dfm, dft;
    double F, p_value;
    ostream & display(ostream &out)const;
  };

  ostream & operator<<(ostream &out, const anova_table &tab);

  Mat add_intercept(const Mat &X);
  Vec add_intercept(const Vec &X);

  //------- virtual base for regression sufficient statistics ----
  class RegSuf: virtual public Sufstat{
  public:
    typedef std::vector<Ptr<RegressionData> > dataset_type;
    typedef Ptr<dataset_type, false> dsetPtr;

    RegSuf * clone()const=0;

    virtual uint size()const=0;  // dimension of beta
    virtual double yty()const=0;
    virtual Vec xty()const=0;
    virtual Spd xtx()const=0;

    virtual Vec xty(const Selector &)const=0;
    virtual Spd xtx(const Selector &)const=0;

    // return least squares estimates of regression params
    virtual Vec beta_hat()const=0;
    virtual double SSE()const=0;  // SSE measured from ols beta
    virtual double SST()const=0;
    virtual double ybar()const=0;
    virtual double n()const=0;

    anova_table anova()const;

    virtual void add_mixture_data(double y, const Vec &x, double prob)=0;
    virtual void combine(Ptr<RegSuf>)=0;

    virtual ostream &print(ostream &out)const;
  };
  inline ostream & operator<<(ostream &out, const RegSuf &suf){
    return suf.print(out);
  }
  //------------------------------------------------------------------
  class QrRegSuf :
    public RegSuf,
    public SufstatDetails<RegressionData>
  {
    mutable QR qr;
    mutable Vec Qty;
    mutable double sumsqy;
    mutable bool current;
  public:
    QrRegSuf(const Mat &X, const Vec &y, bool add_icpt=true);
    QrRegSuf(const QrRegSuf &rhs);  // value semantics

    QrRegSuf *clone()const;
    virtual void clear();
    virtual void Update(const DataType &);
    virtual void add_mixture_data(double y, const Vec &x, double prob);
    virtual uint size()const;  // dimension of beta
    virtual double yty()const;
    virtual Vec xty()const;
    virtual Spd xtx()const;

    virtual Vec xty(const Selector &)const;
    virtual Spd xtx(const Selector &)const;

    virtual Vec beta_hat()const;
    virtual Vec beta_hat(const Vec &y)const;
    virtual double SSE()const;
    virtual double SST()const;
    virtual double ybar()const;
    virtual double n()const;
    void refresh_qr(const std::vector<Ptr<DataType> > &) const ;
    //    void check_raw_data(const Mat &X, const Vec &y);
    virtual void combine(Ptr<RegSuf>);
    virtual void combine(const RegSuf &);
    QrRegSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  };
  //------------------------------------------------------------------
  class NeRegSuf
    : public RegSuf,
      public SufstatDetails<RegressionData>
  {   // directly solves 'normal equations'
  public:
    NeRegSuf(uint p);
    NeRegSuf(const Mat &X, const Vec &y, bool add_icpt=true);
    NeRegSuf(const Spd &xtx, const Vec &xty, double yty, double n);
    template <class Fwd>
    NeRegSuf(Fwd b, Fwd e);
    NeRegSuf(const NeRegSuf &rhs);

    NeRegSuf *clone()const;
    void fix_xtx(bool tf = true);
    virtual void add_mixture_data(double y, const Vec &x, double prob);
    virtual void clear();
    virtual void Update(const RegressionData & rdp);
    virtual uint size()const;  // dimension of beta
    virtual double yty()const;
    virtual Vec xty()const;
    virtual Spd xtx()const;
    virtual Vec xty(const Selector &)const;
    virtual Spd xtx(const Selector &)const;
    virtual Vec beta_hat()const;
    virtual double SSE()const;
    virtual double SST()const;
    virtual double ybar()const;
    virtual double n()const;
    virtual void combine(Ptr<RegSuf>);
    virtual void combine(const RegSuf &);
    NeRegSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;

    // Adding data only updates the upper triangle of xtx_.  This
    // fills in the lower triangle as well, if needed.
    void reflect()const;
  private:
    mutable Spd xtx_;
    mutable bool needs_to_reflect_;
    Vec xty_;
    bool xtx_is_fixed_;
    double sumsqy;
    double n_;
    double sumy_;
  };

  template <class Fwd>
  NeRegSuf::NeRegSuf(Fwd b, Fwd e){
    Ptr<RegressionData> dp = *b;
    uint p = dp->size();
    xtx_ = Spd(p, 0.0);
    xty_ = Vec(p, 0.0);
    sumsqy = 0.0;
    while(b!=e){
      update(*b);
      ++b;
    }
  }


  //------------------------------------------------------------------
  class RegressionDataPolicy
    : public SufstatDataPolicy<RegressionData, RegSuf>
  {
  public:
    typedef RegressionDataPolicy DataPolicy;
    typedef SufstatDataPolicy<RegressionData, RegSuf> DPBase;

    RegressionDataPolicy(Ptr<RegSuf>);
    RegressionDataPolicy(Ptr<RegSuf>, const DatasetType &d);
    template <class FwdIt>
    RegressionDataPolicy(Ptr<RegSuf>, FwdIt Begin, FwdIt End);

    RegressionDataPolicy(const RegressionDataPolicy &);
    RegressionDataPolicy * clone()const=0;
    RegressionDataPolicy & operator=(const RegressionDataPolicy &);

  };
  template <class Fwd>
  RegressionDataPolicy::RegressionDataPolicy(Ptr<RegSuf> s, Fwd b, Fwd e)
    : DPBase(s,b,e)
  {}

  //------------------------------------------------------------------

  class RegressionModel
    : public GlmModel,
      public ParamPolicy_2<GlmCoefs, UnivParams>,
      public RegressionDataPolicy,
      public ConjugatePriorPolicy<RegressionConjSampler>,
      public NumOptModel,
      public EmMixtureComponent
  {
 public:
    RegressionModel(unsigned int p);
    RegressionModel(const Vec &b, double Sigma);

    // the next two constructors are the same, except that the design
    // matrix has information about vnames
    RegressionModel(const Mat &X, const Vec &y, bool add_icpt=true);
    RegressionModel(const DesignMatrix &X, const Vec &y, bool add_icpt=true);
    RegressionModel(const DatasetType &d, bool all=true);

    RegressionModel(const RegressionModel &rhs);
    RegressionModel * clone()const;

    uint nvars()const;  // number of included variables, inc. intercept
    uint nvars_possible()const;  // number of potential variables, inc. intercept

    //---- parameters ----
    Ptr<GlmCoefs> coef();
    const Ptr<GlmCoefs> coef()const;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm()const;

    // beta() and Beta() inherited from GLM;
    //    void set_beta(const Vec &b);
    void set_sigsq(double s2);

    const double & sigsq()const;
    double sigma()const;

    //---- simulate regression data  ---
    virtual RegressionData * simdat()const;
    virtual RegressionData * simdat(const Vec &X)const;
    Vec simulate_fake_x()const;  // no intercept

    //---- estimation ---
    Spd xtx(const Selector &inc)const;
    Vec xty(const Selector &inc)const;
    Spd xtx()const;      // adjusts for covariate inclusion-
    Vec xty()const;      // exclusion, and includes weights,
    double yty()const;   // if used


    void make_X_y(Mat &X, Vec &y)const;

    //--- probability calculations ----
    virtual void mle();
    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
    virtual double pdf(dPtr, bool)const;
    virtual double pdf(const Data *, bool)const;
    double empty_loglike(Vec &g, Mat &h, uint nd)const;

    // directives for how to store data and sufficient statistics
    void use_normal_equations();
    void use_QR();

    void add_mixture_data(Ptr<Data>, double prob);

    void set_conjugate_prior(Ptr<MvnGivenXandSigma>, Ptr<GammaModel>);
    void set_conjugate_prior(Ptr<RegressionConjSampler>);
    //--- diagnostics ---
    Vec hats()const;
    Vec cooks()const;
    Vec VIF()const;
    ostream & print_anova_table(ostream &)const;
    anova_table anova()const{return suf()->anova();}

  };
  //------------------------------------------------------------

}// ends namespace BOOM
#endif
