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
#ifndef BOOM_GLM_COEFS_HPP
#define BOOM_GLM_COEFS_HPP

#include <Models/ParamTypes.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{
  class GlmCoefs
    : public VectorParams
  {
  public:
    typedef std::vector<string> StringVec;
    typedef Ptr<StringVec, false> vnPtr;
    //    typedef boost::signal1<void, const GlmCoefs &> SignalType;
    //    typedef  SignalType::slot_type ObserverType;

    //    GlmCoefs();
    GlmCoefs(uint p, bool all=true);  // beta is 0..p

    // if infer_model_selection is true then zero coefficients will be
    // excluded from the model using drop()
    GlmCoefs(const Vec &b, bool infer_model_selection=false);
    GlmCoefs(const Vec &b, const Selector &Inc);
    GlmCoefs(const GlmCoefs &rhs);
    virtual GlmCoefs *clone()const;

    //---     model selection  -----------
    const Selector &inc()const;
    bool inc(uint p)const;
    void set_inc(const Selector &);
    void add(uint i);
    void drop(uint i);
    void flip(uint i);
    void drop_all();
    void add_all();

    //---- size querries...
    uint size(bool minimal=true)const; // number included/possible covariates
    uint nvars()const;
    uint nvars_possible()const;
    uint nvars_excluded()const;

    //--- the main job of glm's...
    double predict(const Vec &x)const;

    //------ operations for only included variables --------
    Vec beta()const;
    void set_beta(const Vec &b);
    double beta(uint i)const;   // i indexes included covariates
    string vnames(uint i)const; // names of included covariates
    std::vector<string> vnames()const;

    //----- operations for both included and excluded variables ----
    const Vec & Beta()const;    // reports 0 for excluded positions
    void set_Beta(const Vec &, bool reset_inc=true);
    double &  Beta(uint I);        // I indexes possible covariates
    double Beta(uint I)const;        // I indexes possible covariates
    string Vnames(uint I)const;      // names of all potential covariates
    std::vector<string>  Vnames()const;

    void set_vnames(const StringVec &vnames);
    void set_vnames(vnPtr vnames);


    //------  overloads ---------------
    //    virtual istream & read(istream &in);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);

  private:
    Selector inc_;
    vnPtr vnames_;
    mutable Vec beta_;
    mutable bool beta_current_;
    mutable bool Beta_current_;

    void inc_from_beta(const Vec &v);
    uint indx(uint i)const{return inc_.indx(i);}
    void wrong_size_beta(const Vec &b)const;
    //    void watch_beta(const Vec &);
    void fill_beta()const;
    void setup_obs();
    void incompatible_covariates(const Vec &x, const string &fname)const;

    double & operator[](uint i);
    double operator[](uint i)const;
  };

}
#endif// BOOM_GLM_COEFS_HPP
