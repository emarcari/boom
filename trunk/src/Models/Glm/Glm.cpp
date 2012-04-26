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

#include "Glm.hpp"
#include <distributions.hpp>
#include <cpputil/file_utils.hpp>
#include <LinAlg/Types.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace BOOM{


  typedef GlmModel GLM;

  typedef std::vector<string> StringVec;

  //===========================================================

   GlmModel::GlmModel()
   {}

   GlmModel::GlmModel(const GlmModel & rhs)
     : Model(rhs)
   {}

  uint GlmModel::xdim()const{ return coef()->nvars_possible();}
  void GlmModel::add(uint p){ coef()->add(p);}
  void GlmModel::add_all(){ for(int i = 0; i < xdim(); ++i) add(i);}
  void GlmModel::drop(uint p){ coef()->drop(p);}
  void GlmModel::drop_all(){for(int i = 0; i < xdim(); ++i) drop(i);}
  void GlmModel::drop_all_but_intercept(){drop_all();  add(0);}
  void GlmModel::flip(uint p){coef()->flip(p);}
  const Selector & GlmModel::inc()const{return coef()->inc();}
  bool GlmModel::inc(uint p)const{return coef()->inc(p);}

  void GlmModel::set_vnames(const StringVec &vn){
    coef()->set_vnames(vn);}

  double GlmModel::predict(const Vec &x)const{
    return coef()->predict(x);}
  double GlmModel::predict(const VectorView &x)const{
    return coef()->predict(x);}
  double GlmModel::predict(const ConstVectorView &x)const{
    return coef()->predict(x);}

  Vec GlmModel::beta()const{return coef()->beta();}
  void GlmModel::set_beta(const Vec &b){coef()->set_beta(b);}
  const double& GlmModel::beta(int i)const{return beta()[i];}

  // reports 0 for excluded positions
  const Vec & GlmModel::Beta()const{return coef()->Beta();}
  void GlmModel::set_Beta(const Vec &B, bool reset_inc){
    coef()->set_Beta(B, reset_inc);}
  double GlmModel::Beta(uint I)const{return coef()->Beta(I);}
  string GlmModel::Vnames(uint i)const{return coef()->Vnames(i);}
  StringVec GlmModel::Vnames()const{return coef()->Vnames();}

}// closes namespace BOOM;
