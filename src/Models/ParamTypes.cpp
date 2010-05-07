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
#include <stdexcept>
#include <string>
#include <algorithm>

#include <cpputil/io.hpp>
#include <cpputil/file_utils.hpp>
#include "ParamTypes.hpp"
#include <LinAlg/ConstVectorView.hpp>

namespace BOOM{

  Vec vectorize(const ParamVec &v, bool minimal){
    uint N = v.size();
    uint vec_size(0);

    for(uint i=0; i<N; ++i) vec_size+= v[i]->size(minimal);
    Vec ans(vec_size);
    Vec::iterator it=ans.begin();
    for(uint i=0; i<N; ++i){
      Vec tmp=v[i]->vectorize(minimal);
      it = std::copy(tmp.begin(), tmp.end(), it);
    }
    return ans;
  }
  void unvectorize(ParamVec &pvec, const Vec &v, bool minimal){
    Vec::const_iterator it=v.begin();
    for(uint i=0; i<pvec.size(); ++i){
      it = pvec[i]->unvectorize(it, minimal);
    }
  }

  ostream & operator<<(ostream &out, const ParamVec &v){
    out << vectorize(v, false);
    return out;
  }

  Params::Params()
  {}

  Params::Params(const Params &rhs)
    : Data(rhs)
  {}

  void Params::check_io()const{
    if(!io_mgr){
      string msg = "io_manager not set.  Try setting fname first\n";
      throw std::runtime_error(msg);}}

  void Params::output(const Vec &v)const{
    check_io();
    io_mgr->output(v); }

  void Params::input(Vec &v, bool last_line){
    check_io();
    io_mgr->input(v, last_line);
  }

  uint Params::count_lines()const{
    check_io();
    return io_mgr->count_lines();}

  void Params::flush()const{
    check_io();
    io_mgr->flush();  }

  uint Params::io(IO io_prm){

    if(io_prm==WRITE){
      //      output(vectorize(false));
      write();
    }else if(io_prm==CLEAR){
      clear_file();
    }else if(io_prm==READ){
//       Vec v(vectorize(false));
//       input(v, true);
//       unvectorize(v, false);
      read();
    }else if(io_prm==STREAM){
//       Vec v(vectorize(false));
//       input(v,false);
//       unvectorize(v, false);
      stream();
    }else if(io_prm==FLUSH){
      flush();
    }else if(io_prm==COUNT){
      return count_lines();
    }else throw bad_io(io_prm);

    return 0;
  }

  void Params::write()const{
    Vec v(vectorize(false));
    output(v);
  }

  void Params::read(){
    Vec v(vectorize(false));
    input(v);
  }

  void Params::stream(){
    Vec v(vectorize(false));
    input(v,false);
    unvectorize(v,false);
  }

  void Params::set_bufsize(uint p){
    check_io();
    io_mgr->set_bufsize(p);
  }

  void Params::reset_stream(){
    check_io();
    uint dim  = this->size(false);
    io_mgr->reset_stream(dim);
  }

  void Params::clear_file(){
    check_io();
    io_mgr->clear_file();
  }

  void Params::set_fname(const string &fname){
    if(!io_mgr){
      io_mgr.reset(new io_manager(fname));
    }else{
      io_mgr->set_fname(fname);
    }
  }

  void Params::set_io_manager(ParamIoManagerBase * io){
    io_mgr.reset(io);
  }


  //======================================================================

  typedef UnivData<double> UDD;
  UnivParams::UnivParams() : Params(), UDD(0){}
  UnivParams::UnivParams(double x) : UDD(x){}
  UnivParams::UnivParams(const UnivParams&rhs)
    : Data(rhs), Params(rhs), UDD(rhs){}
  UnivParams * UnivParams::clone() const{
    return new UnivParams(*this); }

  Vec UnivParams::vectorize(bool)const{
    Vec ans(1);
    ans[0] = value();
    return ans; }
  Vec::const_iterator UnivParams::unvectorize(Vec::const_iterator &v, bool){
    set(*v);
    return ++v;}
  Vec::const_iterator UnivParams::unvectorize(const Vec &v, bool){
    Vec::const_iterator b=v.begin();
    return unvectorize(b); }

  //============================================================
  typedef VectorData VD;

  VectorParams::VectorParams(uint p, double x)
    : VD(p, x)
  {}

  VectorParams::VectorParams(const Vec &v)
    : VD(v)
  {}

  VectorParams::VectorParams(const VectorParams &rhs)
    : Data(rhs), Params(rhs), VD(rhs)
  {}

  VectorParams * VectorParams::clone()const{
    return new VectorParams(*this);}

//   VectorParams & VectorParams::operator=(const VectorParams &rhs){
//     if(&rhs!=this) VD::operator=(rhs);
//     return *this;
//   }

  Vec VectorParams::vectorize(bool)const{ return value();}

  Vec::const_iterator VectorParams::unvectorize
  (Vec::const_iterator &v, bool){
    Vec::const_iterator e = v+size(false);
    Vec tmp(v,e);
    set(tmp);
    return e;
  }

  Vec::const_iterator VectorParams::unvectorize(const Vec &v, bool){
    Vec::const_iterator b=v.begin();
    return unvectorize(b); }

  //============================================================
  typedef MatrixData MD;
  typedef MatrixParams MP;

  MP::MatrixParams(uint r, uint c, double x)
     : MD(r,c,x)
  {}

  MP::MatrixParams(const Mat &m)
    : MD(m)
  {}

  MP::MatrixParams(const MatrixParams &rhs)
    : Data(rhs), Params(rhs),MD(rhs)
  {}

  MatrixParams * MP::clone()const{
    return new MP(*this);}

//   MatrixParams & MP::operator=(const MatrixParams &rhs){
//     if(&rhs!=this) MD::operator=(rhs);
//     return *this;}

  Vec MP::vectorize(bool)const{
    Vec ans(value().begin(), value().end());
    return ans; }

  Vec::const_iterator MP::unvectorize(Vec::const_iterator &b, bool){
    Vec::const_iterator e=b+size();
    const Mat &val(value());
    Mat tmp(b,e,val.nrow(), val.ncol());
    set(tmp);
    return e;
  }
  Vec::const_iterator MP::unvectorize(const Vec &v, bool){
    Vec::const_iterator b=v.begin();
    return unvectorize(b);
  }

  //============================================================

  typedef CorrParams CP;

  CP::CorrParams(const Corr &R)
    : Params(),
      CorrData(R)
  {}

  CP::CorrParams(const Spd &R)
    : Params(),
      CorrData(R)
  {}

  CP::CorrParams(const CP &rhs)
    : Data(rhs),
      Params(rhs),
      CorrData(rhs)
  {}

  CP * CP::clone()const{return new CP(*this);}

  Vec CP::vectorize(bool minimal)const{
    return value().vectorize(minimal); }

  Vec::const_iterator CP::unvectorize(Vec::const_iterator &v,
				      bool minimal){
    Corr tmp(value());
    Vec::const_iterator ans = tmp.unvectorize(v, minimal);
    set(tmp);
    return ans;
  }

  Vec::const_iterator CP::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator b(v.begin());
    return unvectorize(b,minimal);
  }


}
