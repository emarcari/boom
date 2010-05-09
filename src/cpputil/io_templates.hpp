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

#ifndef BOOM_IO_TEMPLATES_H
#define BOOM_IO_TEMPLATES_H

#include "io.hpp"
#include <iostream>
#include <fstream>
#include <ios>
#include <cpputil/file_utils.hpp> // for gll

namespace BOOM{
  template<class T>
  void io(const std::string &fname, T &obj, IO io_prm, bool ask,
 	  bool newline){
    // check to see if fname is a valid path if not, make it so if ask
    // is true.  else throw exception if ask is false
    typedef unsigned long ulong;

    if(io_prm==CLEAR) clear_file(fname);
    else if(io_prm==FLUSH || io_prm==STREAM){
      // do nothing
    }else if(io_prm==WRITE){
      std::ofstream out(fname.c_str(), std::ios_base::app);
      obj.write(out, newline);
    }else if(io_prm==READ){
      std::ifstream in(fname.c_str());
      gll(in);
      obj.read(in);
    }else throw bad_io(io_prm);
  }

  template<class T>  // specialization for pointers
  inline void io(const std::string &fname, T* &obj, IO io_prm, bool ask,
		 bool newline){
    // check to see if fname is a valid path if not, make it so if ask
    // is true.  else throw exception if ask is false
    typedef unsigned long ulong;

    if(io_prm==CLEAR) clear_file(fname);
    else if(io_prm==FLUSH || io_prm==STREAM){
      // do nothing
    }else if(io_prm==WRITE){
      std::ofstream out(fname.c_str(), std::ios_base::app);
      obj->write(out, newline);
    }else if(io_prm==READ){
      std::ifstream in(fname.c_str());
      gll(in);
      obj->read(in);
    }else throw bad_io(io_prm);
  }

  template<>  // specialization for double
  inline void io(const std::string &fname, double &obj, IO io_prm,
 		 bool, bool newline){
    if(io_prm==CLEAR) clear_file(fname);
    else if(io_prm==FLUSH || io_prm==STREAM){
      // do nothing
    }else if(io_prm==WRITE){
      std::ofstream out(fname.c_str(), std::ios_base::app);
      out << obj << " ";
      if(newline) out <<std::endl;
    }else if(io_prm==READ){
      std::ifstream in(fname.c_str());
      gll(in);
      in >> obj;
    }else throw bad_io(io_prm);
  }


  //-------------- specialization for one double


  template <class T>
  void io(const std::string &fname, T &obj, IO io_prm, bool ask){
    io(fname, obj, io_prm, ask, true); }

  template <class T>
  void io(const std::string &fname, T &obj, IO io_prm){
    io(fname, obj, io_prm, true, true); }

  //----------------------------------------------------------------------


}
#endif // BOOM_IO_TEMPLATES_H


