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
#include "ModelTypes.hpp"
#include <Models/DoubleModel.hpp>
#include <Models/VectorModel.hpp>

#include <algorithm>
#include <TargetFun/Loglike.hpp>
#include <numopt.hpp>
#include <cpputil/ProgressTracker.hpp>

namespace BOOM{

  //    void intrusive_ptr_add_ref(Model *m){ m->up_count(); }
  //    void intrusive_ptr_release(Model *m){
  // m->down_count(); if(m->ref_count()==0) delete m; }

  Model::Model(){}

  Model::Model(const Model &)
    : RefCounted()
  {}

  uint Model::io_params(IO io_prm){
    ParamVec prm(t());
    uint n = prm.size();
    uint ans =0;
    for(uint i=0; i<n; ++i){
      ans = prm[i]->io(io_prm);
      if(io_prm==COUNT) return ans;
    }
    return ans;
  }

  uint Model::count_params()const{
    ParamVec prm(t());
    return prm[0]->count_lines();
  }

  void Model::clear_params()const{
    ParamVec prm(t());
    for(uint i=0; i<prm.size(); ++i){
      prm[i]->clear_file();
    }
  }

  void Model::write_params()const{
    ParamVec prm(t());
    for(uint i=0; i<prm.size(); ++i){
      prm[i]->write();
    }
  }

  void Model::flush_params()const{
    ParamVec prm(t());
    for(uint i=0; i<prm.size(); ++i){
      prm[i]->flush();
    }
  }

  void Model::read_last_params(){
    ParamVec prm(t());
    for(uint i=0; i<prm.size(); ++i){
      prm[i]->read();
    }
  }

  void Model::stream_params(){
    ParamVec prm(t());
    for(uint i=0; i<prm.size(); ++i){
      prm[i]->stream();
    }
  }

  Vec Model::vectorize_params(bool minimal)const{
    ParamVec prm(t());
    uint nprm = prm.size();
    uint N(0), nmax(0);
    for(uint i=0; i<nprm; ++i){
      uint n = prm[i]->size();
      N += n;
      nmax = std::max(nmax, n);
    }
    Vec ans(N);
    Vec wsp(nmax);
    Vec::iterator it = ans.begin();
    for(uint i=0; i<nprm; ++i){
      wsp = prm[i]->vectorize(minimal);
      it = std::copy(wsp.begin(), wsp.end(), it);
    }
    return ans;
  }

  void Model::set_bufsize(uint p){
    ParamVec prm(t());
    uint n = prm.size();
    for(uint i=0; i<n; ++i) prm[i]->set_bufsize(p);}

  void Model::reset_stream(){
    ParamVec prm(t());
    uint n = prm.size();
    for(uint i=0; i<n; ++i) prm[i]->reset_stream();
  }

  void Model::unvectorize_params(const Vec &v, bool minimal){
    ParamVec prm(t());
    Vec::const_iterator b = v.begin();
    for(uint i=0; i<prm.size(); ++i) b = prm[i]->unvectorize(b, minimal);
  }

  uint Model::track_progress(const string & dname, bool restart, uint nskip,
			     const string & prog_name, bool keep_existing_msg){
    progress_.reset(new ProgressTracker(
        dname, nskip, restart, prog_name, keep_existing_msg));
    uint ans = restart ? progress_->restart() : 0 ;
    return ans;
  }

  uint Model::track_progress(uint nskip, const string & prog_name){
    progress_.reset(new ProgressTracker(nskip, prog_name));
    return 0;
  }

  ostream & Model::msg()const{
    if(!progress_){
      ostringstream out;
      out << "message file not set.  Set it with the 'track_progress'"
	  << " model member function." << endl;
      throw_exception<std::logic_error>(out.str());
    }
    return progress_->msg();
  }

  void Model::progress()const{
    if(!!progress_) progress_->update();
  }
  //============================================================
  void MLE_Model::initialize_params(){ mle(); }

  //============================================================
  void LoglikeModel::mle(){
    LoglikeTF loglike(this);
    Vec prms = vectorize_params(true);
    max_nd0(prms, Target(loglike));
    unvectorize_params(prms, true);
  }

  void dLoglikeModel::mle(){
    dLoglikeTF loglike(this);
    Vec prms = vectorize_params(true);
    max_nd1(prms, Target(loglike), dTarget(loglike));
    unvectorize_params(prms, true);
  }


  void d2LoglikeModel::mle(){
    d2LoglikeTF loglike(this);
    Vec prms = vectorize_params(true);
    Vec g(prms);
    uint p = g.size();
    Mat h(p,p);
    max_nd2(prms, g, h, Target(loglike), dTarget(loglike),
	    d2Target(loglike), 1e-5);
    unvectorize_params(prms, true);
  }

  double d2LoglikeModel::mle_result(Vec &g, Mat &h){
    d2LoglikeTF loglike(this);
    Vec prms = vectorize_params(true);
    uint p = prms.size();
    g.resize(p);
    h.resize(p, p);
    double logf = max_nd2(prms, g, h, Target(loglike),
			  dTarget(loglike), d2Target(loglike), 1e-5);
    unvectorize_params(prms, true);
    return logf;
  }

  double DoubleModel::pdf(Ptr<Data> dp, bool logscale)const{
    double x = dp.dcast<DoubleData>()->value();
    double ans = logp(x);
    return logscale?ans : exp(ans);
  }

  double DoubleModel::pdf(const Data * dp, bool logscale)const{
    double x = dynamic_cast<const DoubleData *>(dp)->value();
    double ans = logp(x);
    return logscale?ans : exp(ans);
  }

  //======================================================================
  double DiffDoubleModel::logp(double x)const{
    double g(0),h(0);
    return Logp(x,g,h,0);}
  double DiffDoubleModel::dlogp(double x, double &g)const{
    double h(0);
    return Logp(x,g,h,1);}
  double DiffDoubleModel::d2logp(double x, double &g, double &h)const{
    return Logp(x,g,h,2);}
  //======================================================================
  double DiffVectorModel::logp(const Vec &x)const{
    Vec g;
    Mat h;
    return Logp(x,g,h,0);}
  double DiffVectorModel::dlogp(const Vec &x, Vec &g)const{
    Mat h;
    return Logp(x,g,h,1);}
  double DiffVectorModel::d2logp(const Vec &x, Vec &g, Mat &h)const{
    return Logp(x,g,h,2);}

}
