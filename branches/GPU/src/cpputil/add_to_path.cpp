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

#include <string>
#include <cpputil/file_utils.hpp>

#include "string_utils.hpp"

 namespace BOOM{
   using std::string;

   string add_to_path(const string &path, const char *s){
     return add_to_path(path, string(s));}

   string add_to_path(const char* path, const string &s){
     return add_to_path(string(path),s);}

   string add_to_path(const string &path, const string &s){
     string ans(path);
     if(last(ans)!='/' && path.size()>0) ans+="/";
     ans+=s;
     return ans;
   }


 }
