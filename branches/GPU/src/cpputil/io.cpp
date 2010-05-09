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

#include "io.hpp"
#include "io_templates.hpp"
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/convenience.hpp>
#include <iostream>
#include <stdexcept>
//----------------------------------------------------------------------

namespace BOOM{
  typedef std::string string;
  namespace fs = boost::filesystem;
  using fs::path;

  void clear_file(const string &fname){
    path p(fname);
    if(exists(p)){
      if(is_directory(p)) throw bad_file_name(fname);
    }else{
      path d(get_dpath(fname));
      if(!exists(d)) create_directories(d); }
    fs::ofstream out(p, std::ios_base::trunc);
  }


  void io_raw_data(const string &fname, double *obj, int lo, int hi,
 		   IO io_prm, bool, bool newline){
    if(io_prm==CLEAR) clear_file(fname);
    else if(io_prm==FLUSH){
      // do nothing
    }else if(io_prm==WRITE){
      std::ofstream out(fname.c_str(), std::ios_base::app);
      for(int i=lo; i<=hi; ++i) out << obj[i] << " ";
      if(newline) out <<std::endl;
    }else if(io_prm==READ){
      std::ifstream in(fname.c_str());
      gll(in);
      for(int i =lo; i<=hi; ++i) in >> obj[i];
    }else throw bad_io(io_prm);
  }
  //----------------------------------------------------------------------

  ostream & operator<<(ostream & out, IO io_prm){
    if(io_prm== READ) out << "READ";
    else if(io_prm==WRITE) out << "WRITE";
    else if(io_prm==CLEAR) out << "CLEAR";
    else if(io_prm==FLUSH) out << "FLUSH";
    else if(io_prm==STREAM) out << "STREAM";
    else if(io_prm==COUNT) out << "COUNT";
    else throw std::logic_error("unrecognized io_prm in operator<<");
    return out;
  }

  istream & operator<<(istream & in, IO & io_prm){
    string s;
    in >> s;
    if(s=="READ") io_prm=READ;
    else if(s=="WRITE") io_prm=WRITE;
    else if(s=="CLEAR") io_prm=CLEAR;
    else if(s=="FLUSH") io_prm=FLUSH;
    else if(s=="STREAM") io_prm=STREAM;
    else if(s=="COUNT") io_prm=COUNT;
    else throw std::logic_error("unrecognized io_prm in operator>>");
    return in;
  }

}

