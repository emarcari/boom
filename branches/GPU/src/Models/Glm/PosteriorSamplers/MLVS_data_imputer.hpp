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

//#define CUDA_ENABLED  // Moved to compile command-line options

namespace BOOM{

  namespace mlvs_impute{
    class MDI_base;
    class MDI_worker;
#ifdef CUDA_ENABLED
    class GPU_MDI_worker;
#endif
  }

  class MlvsDataImputer : private RefCounted{
  public:
    MlvsDataImputer(Ptr<MLogitBase> Mod, Ptr<MlvsCdSuf> Suf, uint nthreads, bool useGPU=false);
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

      MDI_worker(Ptr<MLogitBase> mod,
		 Ptr<MlvsCdSuf> s,
		 uint Thread_id=0,
		 uint Nthreads=1);
#ifdef CUDA_ENABLED
      void impute_u(Ptr<ChoiceData> dp);
      uint unmix(double u);
      virtual void operator()();
#else
      void impute_u(Ptr<ChoiceData> dp);
      uint unmix(double u);
      void operator()();
#endif
      const Ptr<MlvsCdSuf> suf()const;
      void seed(unsigned long);
#ifdef CUDA_ENABLED
    protected:
#else
    private:
#endif
      Ptr<MLogitBase> mlm;
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
      MDI_unthreaded(Ptr<MLogitBase> m, Ptr<MlvsCdSuf> s, bool useGPU=false);
      virtual void draw();
    private:
      Ptr<MLogitBase> mlm;
      Ptr<MlvsCdSuf> suf;
      Ptr<MDI_worker> imp;
    };

    //======================================================================
    class MDI_threaded : public MDI_base {
    public:
      MDI_threaded(Ptr<MLogitBase> m, Ptr<MlvsCdSuf> s, uint nthreads);
      virtual void draw();
    private:
      Ptr<MLogitBase> mlm;
      Ptr<MlvsCdSuf> suf;
      std::vector<Ptr<MDI_worker> > crew;
    };

  }
}

#endif// BOOM_MLVS_DATA_IMPUTER_HPP
