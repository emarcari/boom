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

#ifndef BOOM_COMPOSITE_MODEL_HPP
#define BOOM_COMPOSITE_MODEL_HPP
#include "ModelTypes.hpp"
#include "CompositeData.hpp"
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <boost/utility/enable_if.hpp>

namespace BOOM{

  class CompositeModel
    : virtual public Model,
      public CompositeParamPolicy,
      public IID_DataPolicy<CompositeData>,
      public PriorPolicy
  {

    // A composite model assumes that y1, y2, ... are independent with
    // y1 ~ m1, y2 ~ m2, etc.  y1,y2... are stored in CompositeData,
    // and m1, m2, etc. are stored here

  public:
    template <class MOD>
    CompositeModel(const std::vector<Ptr<MOD> > &mod,
		   typename boost::enable_if<
		   boost::is_base_of<Model, MOD>
		   >::type * =0)
      : m_(mod.begin(), mod.end())
    {
      setup();
    }

    CompositeModel(const CompositeModel &rhs);
    CompositeModel * clone()const;

    //    virtual void initialize_params();
    virtual void add_data(Ptr<CompositeData>);
    virtual void add_data(Ptr<Data>);

    double pdf(Ptr<CompositeData>, bool logscale)const;
    double pdf(Ptr<Data>, bool logscale)const;

  protected:

    CompositeModel();

    template <class Fwd>
    void set_models(Fwd b, Fwd e){   // to be called by constructors of
      m_.assign(b,e);                // derived classes
      setup(); }
  private:
    std::vector<Ptr<Model> > m_;
    void setup();
  };

}

#endif // BOOM_COMPOSITE_MODEL_HPP

