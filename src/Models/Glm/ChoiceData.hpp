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

#ifndef BOOM_CHOICE_DATA_HPP
#define BOOM_CHOICE_DATA_HPP

#include <Models/CategoricalData.hpp>
#include <Models/DataTypes.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{

  class ChoiceData : public CategoricalData
  {
  public:

    // Args:
    //   y:  The response
    //   subject_x: A vector of predictors describing subject level
    //     characteristics.  Should contain an intercept.
    //   choice_x: A vector of vectors describing characteristics of
    //     the object being chosen.  May be empty if all data is
    //     subject-level data.
    ChoiceData(const CategoricalData &y,
               const Ptr<VectorData> &subject_x,
               const std::vector<Ptr<VectorData> > & choice_x);

    ChoiceData(const ChoiceData &rhs);

    virtual ChoiceData * clone()const;

    //--------- virtual function over-rides ----

    virtual ostream & display(ostream &)const;
    //    virtual istream & read(istream &);
    virtual uint size(bool minimal=true)const;

    //--------- choice information ----
    uint nchoices()const;     // number of possible choices
    uint n_avail()const;      // number of choices actually available
    bool avail(uint i)const;  // is choice i available to the subject?

    uint subject_nvars()const;
    uint choice_nvars()const;

    const uint & value()const;
    void set_y(uint y);
    const string & lab()const;
    const std::vector<string> & labels()const;
    void set_y(const string &lab);

    const Vec & Xsubject()const;
    const Vec & Xchoice(uint i)const;
    const Vec & x(uint i)const;
    const Mat & write_x(Mat &, bool inc_zero)const;

    virtual const Mat & X(bool include_zero_contstraint=true)const;
    // X(true) returns a matrix of dimension Nchoices by
    // Nchoices*Xsubject.size + Xchoice.size.  X(false) returns a
    // matrix of dimension (Nchoices-1)*Xsubject.size + Xchoice.size.
    // The omitted columns correspond to those coefficients in a
    // choice model which are constrained to zero.

    virtual void set_Xsubject(const Vec &x);
    virtual void set_Xchoice(const Vec &x, uint i);
  private:
    Ptr<VectorData> xsubject_;                // age of car buyer
    std::vector<Ptr<VectorData> > xchoice_;   // price of car

    Selector avail_;                          // which choices are available
    Vec null_;           // zero length.  return for null reference.

    // All subject and choice predictors stretched out into a single
    // predictor vector
    mutable Matrix bigX_;
    mutable Matrix not_so_bigX_;

    mutable bool big_x_current_;
    mutable bool not_so_big_x_current_;

    bool check_big_x(bool include_zeros)const;
  };
  //______________________________________________________________________
}
#endif// BOOM_CHOICE_DATA_HPP
