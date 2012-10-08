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

#ifndef BOOM_MLVS_DATA_IMPUTER_HPP
#define BOOM_MLVS_DATA_IMPUTER_HPP

#include <Models/Glm/ChoiceData.hpp>
#include <Models/Glm/MLogitBase.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_base.hpp>
#include <distributions/rng.hpp>

namespace BOOM{

  namespace mlvs_impute{
    class MDI_base;
    class MDI_worker;
#ifdef CUDA_ENABLED
    class CPU_MDI_worker_original;
    class CPU_MDI_worker_new_flow;
    class CPU_MDI_worker_parallel;
    class CPU_MDI_worker_new_parallel;
    class GPU_MDI_worker;
#endif
  }

  class MlvsDataImputer : private RefCounted{
  public:
    MlvsDataImputer(MLogitBase *Mod, Ptr<MlvsCdSuf> Suf, uint nthreads,
    		int computeMode=0);
    void draw();

    friend void intrusive_ptr_add_ref(MlvsDataImputer *d){
      d->up_count();}
    friend void intrusive_ptr_release(MlvsDataImputer *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

  private:
    Ptr<mlvs_impute::MDI_base> imp;
  };

  //______________________________________________________________________

  namespace mlvs_impute{

    class MDI_worker : private RefCounted {
    public:

      friend void intrusive_ptr_add_ref(MDI_worker *d){d->up_count();}
      friend void intrusive_ptr_release(MDI_worker *d){
	d->down_count(); if(d->ref_count()==0) delete d;}

      MDI_worker(MLogitBase *mod,
		 Ptr<MlvsCdSuf> s,
		 uint Thread_id=0,
		 uint Nthreads=1);
      void impute_u(Ptr<ChoiceData> dp);
      uint unmix(double u);
      const Ptr<MlvsCdSuf> suf()const;
#ifdef CUDA_ENABLED
      virtual void operator()();
#else
      void operator()();
#endif      
      void seed(unsigned long);
#ifdef CUDA_ENABLED
    protected:
#else
    private:
#endif    
      MLogitBase *mlm;
      Ptr<MlvsCdSuf> suf_;

      const uint thread_id;
      const uint nthreads;
      const Vec mu_;        // mean for EV approx
      const Vec sigsq_inv_; // inverse variance for EV approx
      const Vec sd_;        // standard deviations for EV approx
      const Vec logpi_;     // log of mixing weights for EV approx
      const Vec & log_sampling_probs_;
      const bool downsampling_;

      Vec post_prob_;
      Vec u;
      Vec eta;
      Vec wgts;

      boost::shared_ptr<Mat> thisX;
      RNG rng;
    };
    //======================================================================
    class MDI_base : private RefCounted{
    public:
      friend void intrusive_ptr_add_ref(MDI_base *d){d->up_count();}
      friend void intrusive_ptr_release(MDI_base *d){
	d->down_count(); if(d->ref_count()==0) delete d;}

      virtual void draw()=0;
      virtual ~MDI_base(){}
    };

    //======================================================================
    class MDI_unthreaded : public MDI_base {
    public:
      MDI_unthreaded(MLogitBase *m, Ptr<MlvsCdSuf> s, int computeMode=0);
      virtual void draw();
    private:
      MLogitBase *mlm;
      Ptr<MlvsCdSuf> suf;
      Ptr<MDI_worker> imp;
    };

    //======================================================================
    class MDI_threaded : public MDI_base {
    public:
      MDI_threaded(MLogitBase *m, Ptr<MlvsCdSuf> s, uint nthreads);
      virtual void draw();
    private:
      MLogitBase *mlm;
      Ptr<MlvsCdSuf> suf;
      std::vector<Ptr<MDI_worker> > crew;
    };

  }
}

#endif// BOOM_MLVS_DATA_IMPUTER_HPP
