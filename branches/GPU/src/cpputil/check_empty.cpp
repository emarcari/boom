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

#include <iostream>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/operations.hpp>
#include <string>
#include "ask_to_continue.hpp"

namespace BOOM{
using std::cerr;
using std::cin;
using std::endl;
using std::string;

  namespace fs = boost::filesystem;
  void check_empty(const string &dir){
    // checks to see if dir exists and is empty
    bool exists = fs::exists(dir);
    if(!exists) fs::create_directories(dir);
    string msg = dir+
      " contains files which may be overwritten if you continue.";
    ask_to_continue(msg);
  }
}
//======================================================================
