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

#ifndef BOOM_MVN_MODEL_BASE_HPP
#define BOOM_MVN_MODEL_BASE_HPP
#include <Models/ModelTypes.hpp>
#include <Models/VectorModel.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Sufstat.hpp>
#include <Models/DataTypes.hpp>
#include <Models/SpdParams.hpp>

namespace BOOM{

   class MvnSuf: public SufstatDetails<VectorData>{
    public:
     MvnSuf(uint p);
     MvnSuf(double n, const Vec &sum, const Spd &sumsq);
     MvnSuf(const MvnSuf &sf);
     MvnSuf *clone() const;

     void clear();
     void resize(uint p);  // clears existing data
     void Update(const VectorData &x);
     void update_raw(const Vec &x);
     void add_mixture_data(const Vec &x, double prob);

     const Vec & sum()const;
     const Spd & sumsq()const;
     double n()const;
     Vec ybar()const;
     Spd sample_var()const;  // divides by n-1
     Spd var_hat()const;     // divides by n
     Spd center_sumsq(const Vec &mu)const;
     Spd center_sumsq()const;

     void combine(Ptr<MvnSuf>);
     void combine(const MvnSuf &);
     MvnSuf * abstract_combine(Sufstat *s){
      return abstract_combine_impl(this,s); }

     virtual Vec vectorize(bool minimal=true)const;
     virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                             bool minimal=true);
     virtual Vec::const_iterator unvectorize(const Vec &v,
                                             bool minimal=true);
    private:
     Vec sum_;
     mutable Spd sumsq_;     // uncentered
     double n_;              // sample size
     mutable bool sym_;
     void check_symmetry()const;
   };

  //------------------------------------------------------------

  class MvnBase
    : public DiffVectorModel
  {
  public:
    virtual MvnBase * clone()const=0;
    virtual uint dim()const;
    virtual double Logp(const Vec &x, Vec &g, Mat &h, uint nderiv)const;
    virtual const Vec & mu() const=0;
    virtual const Spd & Sigma()const=0;
    virtual const Spd & siginv() const=0;
    virtual double ldsi()const=0;
  };

  //____________________________________________________________
  class MvnBaseWithParams
    : public MvnBase,
      public ParamPolicy_2<VectorParams, SpdParams>,
      public LocationScaleVectorModel
  {
  public:
    MvnBaseWithParams(uint p, double mu=0.0, double sig=1.0);
    // N(mu,V)... if(ivar) then V is the inverse variance.
    MvnBaseWithParams(const Vec &mean, const Spd &V,
		      bool ivar=false);
    MvnBaseWithParams(Ptr<VectorParams>, Ptr<SpdParams>);
    MvnBaseWithParams(const MvnBaseWithParams &);
    MvnBaseWithParams * clone()const=0;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm()const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm()const;

    virtual const Vec & mu() const;
    virtual  const Spd & Sigma()const;
    virtual  const Spd & siginv() const;
    virtual double ldsi()const;

    virtual void set_mu(const Vec &);
    virtual void set_Sigma(const Spd &);
    virtual void set_siginv(const Spd &);
    virtual void set_S_Rchol(const Vec &sd, const Mat &L);
  };


}

#endif// BOOM_MVN_MODEL_BASE_HPP
