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

#include <cpputil/file_utils.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/exception.hpp>

namespace BOOM{
  using std::string;

  namespace fs = boost::filesystem;
  string get_dpath(const string &fname){
    // returns the collection of

    fs::path p(fname);
    if(fs::exists(p) && fs::is_directory(p)) return fname;
    string::size_type n = fname.find_last_of('/');
    if(n==string::npos) return "";
    string ans(fname.substr(0,n));
    return ans;
  }

}

