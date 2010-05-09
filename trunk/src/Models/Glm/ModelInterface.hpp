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

#ifndef BOOM_MODEL_INTERFACE_HPP
#define BOOM_MODEL_INTERFACE_HPP

#include <cpputil/RefCounted.hpp>
#include <string>
#include <vector>
#include <map>
#include <LinAlg/Types.hpp>
#include <iosfwd>
#include <Models/CategoricalData.hpp>
#include <boost/function.hpp>


namespace BOOM{

  namespace ModelFormula{
    class Term{
    };

    class MainEffect{
    };

    class Transformation{
    };

    class CompositeTerm{
    };
  }

  class ModelInterface
    : private RefCounted,
      public LinAlgTypes
  {
    /***********************************************************************
     The job of a ModelInterface is three-fold:

     1) It determines which variables in a data file will be
        included in the model.

     2) Manage transformations of those variables.

     2) It builds the rows of a design matrix from a vector of text
        strings representing one line of the input file.  A
        ModelInterface knows about three types of variables

 	  a) main effects
 	  b) interactions
 	  c) observation indicators.

 	  Any of these can be either continuous or categorical.  A
          missing continuous variable will be augmented with a missing
          data indicator.  A missing categorical variable will take
          the value "missing".

     3) It keeps track of variable names.

     4) It keeps track of missing data.

     5) It keeps track of transformations and spline basis expansions.

     One can interpret a model interface as a mapping from the way
     data are represented in a data file and the way they are
     represented in a model.

     A ModelInterface is initialized by a string with an R-like notation.
    ***********************************************************************/

  public:
    typedef std::string string;
    typedef std::vector<string> Svec;
    typedef std::map<string,uint> Smap;
    typedef unsigned int uint;
    typedef std::ostream ostream;

    friend void intrusive_ptr_add_ref(ModelInterface *m){
      m->up_count(); }
    friend void intrusive_ptr_release(ModelInterface *m){
      m->down_count(); if(m->ref_count()==0) delete m;}

    ModelInterface(const string & mod);
    ModelInterface(const string & mod, const string & fname);

    void scan_data_file(const string &fname);
    void read_data_file(const string &fname);
    void set_field_separator(const string &fs);
    void set_missing_data_string(const string &mis);

    void set_levels(const string &vname, const Svec & levels);
    bool is_missing(const string &var)const;

    Vec build_x(const string & line)const;
    // build_x takes one line of a data file and turns it into one row
    // of a design matrix.

    const std::vector<string> & vnames()const; // vars in conceptual model
    const string & vnames(uint i)const;        // vars in conceptual model
    const Svec & vnames_x()const;              // vars in design matrix
    ostream & print(ostream & out)const;
    ostream & print_model_syntax(ostream & out)const;

  private:
    typedef boost::function<Vec(double)> Trans;
    string missing_data_string_;
    string field_sep_;

    bool add_intercept_;
    Svec                              names_in_file_;
    Svec                              design_names_;
    Smap                              pos_in_file_;
    std::map<string,bool>             is_cont_;
    std::map<string,bool>             missing_;
    std::map<string,Ptr<CatKey> >     cat_levels_;
    std::map<string, Trans>           transformations_;
    std::vector<Svec>                 interactions_;

    //----- for assembling the numerical vector of covariates----
    Vec get_var(const Svec & fields, const string & vname)const;
    Vec transform(const string & vname, const string &val)const;
    uint get_pos_in_file(const string &vname)const;
    Vec make_interaction(const Svec & terms, const Svec & fields)const;
    Vec interact(const Vec &x1, const Vec & x2)const;

    //----- for reading the model structure ------------
    void parse_format_string(const string & fmt);
    bool read_main_effect(const string & line);
    bool read_interaction(const string & line);
    void add_continuous_main_effect(uint pos, const string & name, bool mis);
    void add_categorical_main_effect(uint pos, const string & name,
				     const Svec & levels, bool mis);
    string & strip_comments(string & line)const;
    bool has_star(const string &s)const;
    bool check_missing(const string & s, bool &)const;
    void add_trans(const string & name, const string & args, uint pos);

    //----- error messages ----------------
    void cant_recognize_term(const string &);

    //----- for keeping track of variable characteristics --------
    bool is_cont(const string &vname)const;
    bool is_fully_observed(const string &vname)const;
    Ptr<CatKey> get_key(const string & vname)const;

    //---------- for building variable names -----------
    Svec get_names(const string &vname)const; // levels and obs_flags
    Svec concat(const Svec &s1, const Svec & s2)const;
    Svec int_names(const Svec & inter, const Svec & csb)const;
    Svec string_int(const Svec & s1, const Svec & s2)const;
    void make_vnames_x(const std::vector<double> &knots);

    //---------- for transformations -------------------
    void add_log(const string &arg, uint pos);
    void add_sqrt(const string &arg, uint pos);
    void add_spline(const string &arg, uint pos);

  };
  //______________________________________________________________________
  class RegressionModelInterface
    : public ModelInterface
  {
  public:
    double build_y(const Svec & fields)const;
  private:
    uint ypos;
  };
  //______________________________________________________________________
  class MulRegInterface
    : public ModelInterface
  {
  public:
    Vec build_y(const Svec & fields)const;
  private:
    std::vector<uint> ypos;
  };

  //______________________________________________________________________
  class ChoiceModelInterface
    : public ModelInterface
  {
  public:
    Ptr<CategoricalData> build_y(const Svec & fields)const;
  private:
    uint ypos;
    Ptr<CatKey> y_levels_;
  };


}

#endif// BOOM_MODEL_INTERFACE_HPP
