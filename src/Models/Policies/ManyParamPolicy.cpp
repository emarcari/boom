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
#include "ManyParamPolicy.hpp"
namespace BOOM{
  typedef ManyParamPolicy MPP;

  MPP::ManyParamPolicy(){}

  MPP::ManyParamPolicy(const MPP &rhs)
    : Model(rhs)
  {}

  ManyParamPolicy & MPP::operator=(const ManyParamPolicy &rhs){
    if(&rhs!=this){
      clear();
      setup_params();
    }
    return *this;
  }

  void MPP::add_params(Ptr<Params> p){t_.push_back(p);}
  void MPP::clear(){t_.clear();}

  ParamVec MPP::t(){return t_;}
  const ParamVec MPP::t()const{return t_;}

  void MPP::set_param_filenames(const std::vector<string> &fnames){
    assert(t_.size()==fnames.size());
    for(uint i=0; i<fnames.size(); ++i){
      t_[i]->set_fname(fnames[i]);
    }
  }

}
