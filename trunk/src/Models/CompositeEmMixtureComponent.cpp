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

#include "CompositeEmMixtureComponent.hpp"

namespace BOOM{

  typedef CompositeEmMixtureComponent CME;
  typedef CompositeModel CM;

  CME::CompositeEmMixtureComponent() {}

  CME::CompositeEmMixtureComponent(const CME &rhs)
    : Model(rhs),
      CompositeModel(rhs.m_),
      EmMixtureComponent(rhs)
  {
    uint S = rhs.m_.size();
    for(uint s=0; s<S; ++s) m_.push_back(rhs.m_[s]->clone());
    CM::set_models(m_.begin(), m_.end());
  }

  CME * CME::clone()const{return new CME(*this);}

  void CME::mle(){
    for(int s = 0; s < m_.size(); ++s){
      m_[s]->mle();
    }
  }

  void CME::find_posterior_mode(){
    for(uint s=0; s<m_.size(); ++s){
      m_[s]->find_posterior_mode();
    }
  }

  void CME::add_mixture_data(Ptr<Data> dp, double prob){
    Ptr<CompositeData> d(CM::DAT(dp));
    uint S = m_.size();
    assert(d->dim() == S);
    for(uint s=0; s<S; ++s) m_[s]->add_mixture_data(d->get_ptr(s), prob);
  }

  void CME::add_model(Ptr<EmMixtureComponent> new_model){
    m_.push_back(new_model);
    CM::add_model(new_model);
  }
}
