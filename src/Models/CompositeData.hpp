/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_COMPOSITE_DATA_HPP
#define BOOM_COMPOSITE_DATA_HPP


#include <Models/DataTypes.hpp>
namespace BOOM{

  class CompositeData : virtual public Data{
  public:
    CompositeData(const std::vector<Ptr<Data> > &d);
    CompositeData * clone()const;

    virtual ostream & display(ostream &)const;
    //    virtual istream & read(istream &);
    virtual uint size(bool minimal=true)const;
    uint dim()const;
    Ptr<Data> get(uint n);
  private:
    std::vector<Ptr<Data> > dat;
  };

}

#endif // BOOM_COMPOSITE_DATA_HPP
