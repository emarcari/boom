/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_MVREG_HPP
#define BOOM_MVREG_HPP
#include <Models/Sufstat.hpp>
#include <Models/SpdParams.hpp>
#include <Models/Glm/Glm.hpp>
#include <LinAlg/QR.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <boost/bind.hpp>

namespace BOOM{

  class MvRegSuf : virtual public Sufstat{
  public:
    typedef std::vector<Ptr<MvRegData> > dataset_type;
    typedef Ptr<dataset_type, false> dsetPtr;

    MvRegSuf * clone()const=0;

    uint xdim()const;
    uint ydim()const;
    virtual const Spd & yty()const=0;
    virtual const Mat & xty()const=0;
    virtual const Spd & xtx()const=0;
    virtual double n()const=0;
    virtual double sumw()const=0;

    virtual Spd SSE(const Mat &B)const=0;

    virtual Mat beta_hat()const=0;
    virtual void combine(Ptr<MvRegSuf>)=0;
  };
  //------------------------------------------------------------
  class MvReg;
  class QrMvRegSuf
    : public MvRegSuf,
      public SufstatDetails<MvRegData>
  {
  public:
    QrMvRegSuf(const Mat &X, const Mat &Y, MvReg *);
    QrMvRegSuf(const Mat &X, const Mat &Y, const Vec &w, MvReg *);
    QrMvRegSuf * clone()const;

    virtual void Update(const MvRegData &);
    virtual Mat beta_hat()const;
    virtual Spd SSE(const Mat &B)const;
    virtual void clear();

    virtual const Spd & yty()const;
    virtual const Mat & xty()const;
    virtual const Spd & xtx()const;
    virtual double n()const;
    virtual double sumw()const;

    void refresh(const std::vector<Ptr<MvRegData> > &)const;
    void refresh(const Mat &X, const Mat &Y)const;
    void refresh(const Mat &X, const Mat &Y, const Vec &w)const;
    void refresh()const;
    virtual void combine(Ptr<MvRegSuf>);
    virtual void combine(const MvRegSuf &);
    QrMvRegSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    mutable QR qr;
    mutable Mat y_;
    mutable Vec w_;

    MvReg * owner;

    mutable bool current;
    mutable Spd yty_;
    mutable Spd xtx_;
    mutable Mat xty_;
    mutable double n_;
    mutable double sumw_;
  };

  //------------------------------------------------------------
  class NeMvRegSuf
    : public MvRegSuf,
      public SufstatDetails<MvRegData>
  {
  public:
    NeMvRegSuf(uint xdim, uint ydim);
    NeMvRegSuf(const Mat &X, const Mat &Y, bool add_int=false);
    template <class Fwd> NeMvRegSuf(Fwd b, Fwd e);
    NeMvRegSuf(const NeMvRegSuf &rhs);
    NeMvRegSuf * clone()const;

    virtual void clear();
    virtual void Update(const MvRegData & rdp);
    virtual void update_raw_data(const Vec &Y, const Vec &X, double w=1.0);
    virtual Mat beta_hat()const;

    virtual Spd SSE(const Mat &B)const;
    virtual const Spd & yty()const;
    virtual const Mat & xty()const;
    virtual const Spd & xtx()const;
    virtual double n()const;
    virtual double sumw()const;
    virtual void combine(Ptr<MvRegSuf>);
    virtual void combine(const MvRegSuf &);
    NeMvRegSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    Spd yty_;
    Spd xtx_;
    Mat xty_;
    double sumw_;
    double n_;
  };

  template <class Fwd>
  NeMvRegSuf::NeMvRegSuf(Fwd b, Fwd e)
  {
    Ptr<MvRegData> dp =*b;
    const Vec &x(dp->x());
    const Vec &y(dp->y());

    uint xdim= x.size();
    uint ydim= y.size();
    xtx_ = Spd(xdim, 0.0);
    yty_ = Spd(ydim, 0.0);
    xty_ = Mat(xdim, ydim, 0.0);
    n_ = 0;

    while(b!=e){ this->update(*b); ++b; }
  }


  //============================================================

  class MvReg
    : public ParamPolicy_2<MatrixParams,SpdParams>,
      public SufstatDataPolicy<MvRegData, MvRegSuf>,
      public PriorPolicy,
      public LoglikeModel
  {
  public:

    MvReg(uint xdim, uint ydim);
    MvReg(const Mat &X, const Mat &Y, bool add_int=false);
    MvReg(const Mat &B, const Spd &Sigma);

    MvReg(const MvReg & rhs);
    MvReg * clone()const;

    uint xdim()const;  // x includes intercept
    uint ydim()const;

    const Mat & Beta()const;     // [xdim rows, ydim columns]
    const Spd & Sigma()const;
    const Spd & Siginv()const;
    double ldsi()const;

    // access to parameters
    Ptr<MatrixParams> Beta_prm();
    const Ptr<MatrixParams> Beta_prm()const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm()const;

    void set_Beta(const Mat &B);
    void set_Sigma(const Spd &V);
    void set_Siginv(const Spd &iV);

    //    void make_X_Y(Mat &X, Mat &Y)const;

    //--- estimation and probability calculations
    virtual void mle();
    virtual double loglike()const;
    virtual double pdf(dPtr,bool)const;
    virtual Vec predict(const Vec &x)const;


    //---- simulate MV regression data ---
    virtual MvRegData * simdat()const;
    virtual MvRegData * simdat(const Vec &X)const;
    Vec simulate_fake_x()const;  // no intercept

  };
}
#endif // BOOM_MVREG_HPP
