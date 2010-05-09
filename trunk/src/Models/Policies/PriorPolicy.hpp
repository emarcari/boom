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

#ifndef BOOM_PRIOR_POLICY_HPP
#define BOOM_PRIOR_POLICY_HPP

#include <Models/ModelTypes.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <map>

namespace BOOM{

  class PriorPolicyBase
      : private RefCounted
  {
   public:
    virtual  PriorPolicyBase * clone()const=0;
    virtual void sample_posterior()=0;
    virtual double logpri()const=0;
    virtual void set_method(Ptr<PosteriorSampler>)=0;
    friend void intrusive_ptr_add_ref(PriorPolicyBase *d){d->up_count();}
    friend void intrusive_ptr_release(PriorPolicyBase *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

   private:

  };



  class PriorPolicy : virtual public Model{
    // policy class to cover how a model sets priors for its paramters.
  public:
    virtual PriorPolicy * clone()const=0;

    virtual void sample_posterior();  //
    virtual double logpri()const;
    virtual void set_method(Ptr<PosteriorSampler>);

    virtual void clear_methods();

  private:
    std::vector<Ptr<PosteriorSampler> > samplers_;
  };
}
#endif // BOOM_PRIOR_POLICY_HPP
