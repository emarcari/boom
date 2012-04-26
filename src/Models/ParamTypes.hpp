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

#ifndef BOOM_PARAM_TYPES_H
#define BOOM_PARAM_TYPES_H

#include "DataTypes.hpp"
#include <cpputil/io.hpp>
#include <boost/scoped_ptr.hpp>

namespace BOOM{

  class Params : virtual public Data{
    // abstract base class.  Params inherit from data so that the
    // parameters of one level in a hierarchical model can be viewed as
    // data for the next.

    // Params have certain IO capabilities that ordinary Data do not
    // have.  Params can use the 'io' member function with arguments
    // 'WRITE'  stores the current value to a parameter history file
    // 'READ'   reads in the last stored value, useful for restarting
    // 'STREAM' reads in the parameter history file sequentially
    // 'COUNT'  returns the number of parameters stored in the history
    //          file
    // 'FLUSH'  parameter output is buffered for efficiency.  calling
    //          FLUSH empties the buffer
  public:

    //---------- construction, assignment, operator=/== ---------
    Params();
    Params(const Params &rhs);  // does not copy io buffer
    virtual ~Params(){}
    virtual Params *clone() const =0;
    // copied/cloned params have distinct data and distinct io buffers

    // Params can be 'vectorized' which allows for coherent io and
    // serialization
    virtual uint size(bool minimal = true)const=0;
    virtual Vec vectorize(bool minimal=true)const=0;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true)=0;
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true)=0;

    // io functions. vectorized params are written to fname through an
    // io buffer

    virtual uint io(IO io_prm);  // deprecated

    void set_bufsize(uint p); // sets size of input-output buffer
    void reset_stream();      // sets the stream back to the beginning

    void set_fname(const string &fname);
    // 'read' and 'display' inherited from Data

    void set_io_manager(ParamIoManagerBase *);  // assumes ownership
    ParamIoManagerBase * get_io_manager(){return io_mgr.get();}

    void write()const;
    uint count_lines()const;
    void flush()const;
    void clear_file();
    void read();
    void stream();

  protected:
    void output(const Vec &v)const;
    void input(Vec &v, bool last_line=true);
  private:
    void check_io()const;
    mutable boost::scoped_ptr<ParamIoManagerBase> io_mgr;
  };

  //============================================================
  //---- non-member functions for vectorizing lots of params ----
  typedef std::vector<Ptr<Params> > ParamVec;

  Vec vectorize(const ParamVec &v, bool minimal=true);
  void unvectorize(ParamVec &pvec, const Vec &v, bool minimal=true);

  ostream & operator<<(ostream &out, const ParamVec &v);

  //============================================================

  class UnivParams : virtual public Params,
		     public DoubleData
  {
  public:
    UnivParams();
    UnivParams(double x);
    UnivParams(const UnivParams &rhs);
    UnivParams * clone() const;


    virtual uint size(bool=true)const {return 1;}
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
  private:
  };

  //------------------------------------------------------------
  class VectorParams : public VectorData,
		       virtual public Params
  {
  public:
    explicit VectorParams(uint p, double x=0.0);
    VectorParams(const Vec &v);  // copies v's data
    VectorParams(const VectorParams &rhs); // copies data
    VectorParams * clone()const;

    virtual uint size(bool minimal=true)const {return VectorData::size(minimal);}
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
  private:
  };
  //------------------------------------------------------------
  class MatrixParams : public MatrixData,
		       virtual public Params
  {
  public:
    MatrixParams(uint r, uint c, double x=0.0);  // zero matrix
    MatrixParams(const Mat &m);  // copies m's data
    MatrixParams(const MatrixParams &rhs); // copies data
    MatrixParams * clone()const;

    virtual uint size(bool minimal=true)const{return MatrixData::size(minimal);}
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
  private:
    //    MatrixParams & operator=(const MatrixParams &rhs);
  };
  //------------------------------------------------------------
  class CorrParams : public CorrData, virtual public Params{
  public:
    CorrParams(const Corr &y);
    CorrParams(const Spd &y);
    CorrParams(const CorrParams &rhs);
    CorrParams * clone()const;

    virtual uint size(bool minimal = true)const{
      return CorrData::size(minimal);}
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
  };


}
#endif //  BOOM_PARAM_TYPES_H
