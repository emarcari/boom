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
#include "SubjectSliceSampler.hpp"
#include <Models/IRT/Subject.hpp>
#include <Models/IRT/SubjectPrior.hpp>

#include <Samplers/SliceSampler.hpp>

#include <cpputil/ParamHolder.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{
  namespace IRT{

    typedef SubjectTF TF;
    TF::SubjectTF(Ptr<Subject> s, Ptr<SubjectPrior> p)
      : sub(s),
	pri(p),
	prms(sub->Theta_prm()),
	wsp(sub->Theta())
    {}

    double TF::operator()(const Vec &v)const{
      ParamHolder ph(v, prms, wsp);
      double ans = pri->pdf(sub, true);
      if(ans==BOOM::negative_infinity()) return ans;
      ans+= sub->loglike();
      return ans;
    }

    //======================================================================

    typedef SubjectSliceSampler SSS;
    SSS::SubjectSliceSampler(Ptr<Subject> s, Ptr<SubjectPrior> p)
      : sub(s),
	pri(p),
	target(sub, pri),
	sam(new SliceSampler(target))
    { }

    SSS * SSS::clone()const{return new SSS(*this);}

    void SSS::draw(){
      Theta = sam->draw(sub->Theta());
      sub->set_Theta(Theta);
    }

    double SSS::logpri()const{ return pri->pdf(sub, true);}

  }
}
