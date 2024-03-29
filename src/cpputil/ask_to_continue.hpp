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

#ifndef BOOM_ASK_TO_CONTINUE_HPP
#define BOOM_ASK_TO_CONTINUE_HPP

#include <iostream>
#include <string>
namespace BOOM{
  inline bool ask_to_continue(const std::string & msg="",
			      std::ostream &out = std::cerr,
			      std::istream &in = std::cin){

    out << msg << " Continue? [y/n] ";
    char c='a';
    in >> c;
    if(c=='n' || c=='N'){
      out << std::endl;
      exit(0);
    }
    if(c=='y' || c=='Y'){
      out << std::endl;
      return true;
    }
    else out << "Please answer y or n." << std::endl;
    return ask_to_continue(msg, out, in);
  }
}
#endif // BOOM_ASK_TO_CONTINUE_HPP
