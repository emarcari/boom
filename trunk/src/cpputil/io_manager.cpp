#include "io.hpp"
#include <cpputil/file_utils.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>

namespace BOOM{
  using namespace std;
  typedef io_manager_base IOMB;
  typedef output_manager OM;
  typedef input_manager IM;

  IOMB::io_manager_base()
      : fname_(""),
      page_size(1000)
  {}

  IOMB::io_manager_base(const string &Fname, uint p)
    : fname_(Fname),
      page_size(p){}

  IOMB::~io_manager_base()
  {
    //    cerr << "in destructor for io_manager_base (base class)" << endl;
  }

  void IOMB::set_fname(const string &s){fname_=s;}
  uint IOMB::page(uint n){ std::swap(n, page_size); return n;}
  uint IOMB::page()const{return page_size;}
  uint IOMB::buf_size()const{return buf.size();}
  bool IOMB::empty()const{return buf.empty();}
  const string & IOMB::file()const{return fname_;}
  //======================================================================
  //  OM::output_manager(){}
  OM::output_manager(const string &Fname, uint p)
    : io_manager_base(Fname, p){}

  OM::~output_manager(){
    //    cerr << "in destructor for output_manager" << endl;
    if(!empty()) write();
  }

 void OM::clear_file()const{ BOOM::clear_file(fname_); }

  void OM::write(){
    ofstream out(fname_.c_str(), ios_base::app);
    while(!empty()) pop().write(out, true); }

  Vec OM::pop(){
    Vec p = buf.back();
    buf.pop_back();
    return p;}

  void OM::push(const Vec & p){
    buf.push_front(p);  // make fresh copy
    if(buf_size()>=page_size) write();
  }
  void OM::set_fname(const string &fn){
    if(fn!=fname_){
      write();
      io_manager_base::set_fname(fn);}}
  //======================================================================
//   IM::input_manager(uint vec_sz)
//     : prototype(vec_sz)
//   {}
  IM::input_manager(uint vec_sz, const string &fn, uint p)
    : io_manager_base(fn, p),
      prototype(vec_sz),
      first_time(true)
  {}

  IM::~input_manager(){}

  void IM::set_prototype(uint vec_size){prototype=Vec(vec_size);}
  uint IM::vec_size()const{return prototype.size();}

  void IM::reset_stream(){
    first_time=true;
    buf.clear();
  }

  void IM::read(){
    ifstream in(fname_.c_str());
    if(!in){
      ostringstream err;
      err << "IoManager cannot read from file " << endl
          << fname_ << endl;
      throw_exception<std::runtime_error>(err.str());
    }

    if(!first_time){
      // file has been read before
      in.seekg(last_read_pos);
    }
    for(uint i=0; i<page_size; ++i){
      Vec p(prototype);
      p.read(in);
      if(in){
	push(p);
      }else{
	break;
      }
    }
    last_read_pos = in.tellg();
    first_time=false;
  }

  Vec IM::read_last(){
    ifstream in(fname_.c_str());
    gll(in);
    prototype.read(in);
    last_read_pos = in.tellg();
    first_time=false;
    return prototype;
  }

  void IM::push(const Vec &p){ buf.push_front(p); }

  Vec IM::pop(){
    if(empty()) read();
    if(empty()){
      string msg = "could not read from file "  + fname_;
      throw_exception<std::runtime_error>(msg);
    }
    Vec ans(buf.back());
    buf.pop_back();
    return ans;
  }
  void IM::set_fname(const string &fn){
    buf.clear();
    io_manager_base::set_fname(fn);
  }
  //======================================================================
  typedef io_manager IOM;

  IOM::io_manager(const string &Fname, uint Bufsize)
    : fname_(Fname),
      bufsize_(Bufsize)
  {}

  void IOM::output(const Vec &v){
    if(!out){
      out.reset(new output_manager(fname_, bufsize_));
      out->set_fname(fname_);
    }
    out->push(v);
  }

  void IOM::input(Vec &v, bool last_line){
    if(!in || in->file()!=fname_){
      in.reset(new input_manager(v.size(), fname_, bufsize_));
      in->set_fname(fname_);
    }
    v= last_line ? in->read_last() : in->pop();
  }

  uint IOM::count_lines()const{
    return BOOM::count_lines(fname_);
  }

  uint IOM::bufsize()const{ return bufsize_; }

  void IOM::set_bufsize(uint n){
    bufsize_ = n;
    if(!!out) out->page(n);
    if(!!in) in->page(n);
  }

  void IOM::reset_stream(uint vsize){
    if(!in){
      in.reset(new input_manager(vsize, fname_, bufsize_));
      in->set_fname(fname_);
    }
    if(!!in) in->reset_stream();
    else{
      ostringstream err;
      err << "input manager has not been set in IOM::reset_stream()";
      throw_exception<std::runtime_error>(err.str());
    }
  }

  void IOM::set_fname(const string &fname){
    fname_ = fname;
    if(!!out) out->set_fname(fname_);
    if(!!in) in->set_fname(fname_);
  }

  void IOM::clear_file(){
    if(!out) out.reset(new output_manager(fname_, bufsize_));
    out->set_fname(fname_);
    out->clear_file();
  }

  void IOM::flush(){ if(!!out) out->write();}
}
