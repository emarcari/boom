/*
  Copyright (C) 2007 Steven L. Scott

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

#include "CompositeData.hpp"

namespace BOOM{

  typedef CompositeData CD;
  CD::CompositeData(const std::vector<Ptr<Data> > &d)
    : dat(d)
  {}

  CD * CD::clone()const{return new CD(*this);}

  ostream & CD::display(ostream &out)const{
    uint n = dat.size();
    for(uint i=0; i<n; ++i) dat[i]->display(out) << " ";
    return out;
  }

//   istream & CD::read(istream & in){
//     uint n = dat.size();
//     for(uint i=0; i<n ;++i) dat[i]->read(in);
//     return in;
//   }

  uint CD::size(bool minimal)const{
    uint ans=0;
    uint n = dat.size();
    for(uint i=0; i<n; ++i) ans += dat[i]->size(minimal);
    return ans;
  }

  uint CD::dim()const{ return dat.size();}

  Ptr<Data> CD::get(uint i){return dat[i];}
}
