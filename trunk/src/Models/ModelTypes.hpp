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
#ifndef BOOM_MODEL_TYPES_HPP
#define BOOM_MODEL_TYPES_HPP

#include <BOOM.hpp>
#include "ParamTypes.hpp"
#include <LinAlg/Types.hpp>
#include <boost/shared_ptr.hpp>

namespace BOOM{

  class PosteriorSampler;
  class Params;
  class Data;
  class ProgressTracker;

  // to inherit from Model a class must provide:
  // clone()   (covariant return)
  // double pdf(Ptr<Data>, bool)const


// a Model is the basic unit of operation in statistical learning.  A
// Model manages Params, Data, and learning methods.

  // a Model should also inherit from a ParamPolicy, a DataPolicy, and
  // a PriorPolicy


  class Model : private RefCounted{
  public:
    friend void intrusive_ptr_add_ref(Model *d){d->up_count();}
    friend void intrusive_ptr_release(Model *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

    //------ constructors, destructors, operator=/== -----------
    Model();
    Model(const Model &rhs);        // ref count is not copied
    virtual Model * clone()const=0;

    // the result of clone() should have identical parameters in
    // distinct memory.  It should not have any data assigned.  Nor
    // should it include the same priors and sampling methods

    virtual ~Model(){}

    //----------- parameter interface  ---------------------
    virtual ParamVec t()=0;              // over-ridden in ParmPolicy
    virtual const ParamVec t()const=0;
    //    virtual void initialize_params()=0;  // MLE_Models over-ride this

    virtual uint count_params()const;
    virtual void clear_params()const;
    virtual void write_params()const;
    virtual void flush_params()const;
    virtual void read_last_params();
    virtual void stream_params();
    virtual uint io_params(IO io_prm);  // deprecated

    virtual Vec vectorize_params(bool minimal=true)const;
    virtual void unvectorize_params(const Vec &v, bool minimal=true);
    virtual void set_bufsize(uint p);
    virtual void reset_stream();

    //------------ functions over-ridden in DataPolicy  -----
    virtual void add_data(Ptr<Data>)=0;    //
    virtual void clear_data()=0;    //
    virtual void combine_data(const Model & , bool just_suf=true)=0;

    //------------ functions over-ridden in PriorPolicy ----
    virtual void sample_posterior()=0;
    virtual double logpri()const=0;      // evaluates current params
    virtual void set_method(Ptr<PosteriorSampler>)=0;

    //--------------------------
    void progress()const;
    uint track_progress(const string &histdir, bool restart=false,
			uint nskip = 100, const string &prog_name="",
                        bool keep_existing_msg=false);
    uint track_progress(uint nskip = 100, const string &prog_name="");
    ostream & msg()const;

  private:
    boost::shared_ptr<ProgressTracker> progress_;
  };

  //============= mix-in classes =========================================
  class MLE_Model: virtual public Model{
  public:
    // model that can be estimated by maximum likelihood
    virtual void mle()=0;
    virtual void initialize_params();  //sets params to MLE
    virtual MLE_Model *clone()const=0;
  };

  class LoglikeModel : virtual public MLE_Model{
  public:
    virtual double loglike()const=0;
    virtual LoglikeModel * clone()const=0;
    virtual void mle();
  };

  class dLoglikeModel : public LoglikeModel{
  public:
    virtual double dloglike(Vec &g)const=0;
    virtual void mle();
    virtual dLoglikeModel *clone()const=0;
  };

  class d2LoglikeModel : public dLoglikeModel{
  public:
    virtual double d2loglike(Vec &g, Mat &H)const=0;
    virtual void mle();
    virtual double mle_result(Vec &gradient, Mat &hessian);
    virtual d2LoglikeModel *clone()const=0;
  };
  class NumOptModel : public d2LoglikeModel{
  public:
    virtual double Loglike(Vec &g, Mat &H, uint nd)const=0;
    virtual double loglike()const{ Vec g; Mat h; return Loglike(g,h,0);}
    virtual double dloglike(Vec &g)const{Mat h; return Loglike(g,h,1);}
    virtual double d2loglike(Vec &g, Mat &h)const{return Loglike(g,h,2);}
    virtual NumOptModel * clone()const=0;
  };
  //======================================================================
  class LatentVariableModel : virtual public Model{
  public:
    virtual void impute_latent_data()=0;
    //    virtual double complete_data_loglike()const=0;
    virtual LatentVariableModel * clone()const=0;
  };
  //======================================================================
  class CorrModel
    : virtual public Model{
  public:
    virtual CorrModel * clone()const=0;
    virtual double logp(const Corr &)const=0;
  };
  //======================================================================
  class MixtureComponent
      : virtual public Model{
   public:
    //    virtual double pdf(Ptr<Data>, bool logscale)const=0;
    virtual double pdf(const Data *, bool logscale)const=0;
    virtual MixtureComponent * clone()const=0;
  };

}
#endif // BOOM_MODEL_TYPES_HPP
