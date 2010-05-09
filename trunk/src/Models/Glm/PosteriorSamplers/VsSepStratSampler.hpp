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

#ifndef BOOM_VS_SEP_STRAT_SAMPLER
#define BOOM_VS_SEP_STRAT_SAMPLER

#include <Models/Glm/GlmMvnPrior.hpp>
#include <Models/GammaModel.hpp>
#include <TargetFun/TargetFun.hpp>
#include <vector>

namespace BOOM{
  class VsSepStratSampler
    : public PosteriorSampler{
  public:
    VsSepStratSampler(Ptr<GlmMvnPrior> mod,
		      std::vector<Ptr<GammaModel> > S_pri);
    virtual void draw();
    virtual double logpri()const;
    uint dim()const;
  private:
    Ptr<GlmMvnPrior>  mod;
    std::vector<Ptr<GammaModel> > Spri;
    Vec Sinv;
    Mat L;
    Mat LT;
    Spd Sumsq;
    Vec nobs;

    void compute_suf();
    void modify_sumsq_with_Sinv();

    void draw_Sinv();
    void draw_L();
    void draw_L(uint i, uint j);
    void set_Sigma();
    double compute_L_limit(uint i, uint j);
    void observe_sigma(const Spd &);

    mutable bool L_current;

    //----------------------
    class SinvTF : public ScalarTargetFun{
    public:
      SinvTF(const Spd & Sumsq, double nobs, const Spd & Rinv,
	     const Vec & Sinv, uint which, Ptr<GammaModel> pri);
      double operator()(double Sinv_i)const;
      //      SinvTF * clone()const{return new SinvTF(*this);}
    private:
      const Spd & Sumsq_;
      const double nobs_;
      const Spd & Rinv_;
      const Vec & Sinv_vec_;
      uint which_;
      Ptr<GammaModel> prior_;
    };
    //-------------------------------------------------
    class LTF : public ScalarTargetFun{
      typedef std::vector<Ptr<GlmCoefs> > DataSet;
    public:
      LTF();
      LTF(const Spd & sumsq, uint i, uint j, Mat &L,
	  Mat &wsp, const DataSet &dat);
      //      LTF * clone()const{return new LTF(*this);}
      double operator()(double Lij)const;
      double limit()const;
    private:
      double eval_pri()const;
      double eval_det()const;
      double eval_qform()const;
      double eval_logdet_L(Mat & wsp, const Selector &)const;
      void set_value(double Lij)const;

      const Spd & Sumsq;
      uint el_i,el_j;
      mutable Mat &Ltrans;
      mutable Mat &L;
      mutable Mat &wsp;
      mutable Spd & Rinv;
      const std::vector<Ptr<GlmCoefs> > &dat;
    };
    //------------------------------------------------
  };
}
#endif // BOOM_VS_SEP_STRAT_SAMPLER
