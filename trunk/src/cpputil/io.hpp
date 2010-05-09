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

#ifndef BOOM_IO_H
#define BOOM_IO_H

#include <string>
#include <iosfwd>
#include <BOOM.hpp>
#include <deque>
#include <LinAlg/Types.hpp>
#include <LinAlg/Vector.hpp>
#include <boost/scoped_ptr.hpp>

namespace BOOM{

  class ParamIoManagerBase{
   public:
    virtual ~ParamIoManagerBase(){}
    virtual uint bufsize()const=0;
    virtual void set_bufsize(uint)=0;
    virtual void reset_stream(uint)=0;
    virtual void set_fname(const string &)=0;
    virtual void output(const Vec &)=0;
    virtual void input(Vec &v, bool last_line=true)=0;
    virtual uint count_lines()const=0;
    virtual void clear_file()=0;
    virtual void flush()=0;
    virtual const string & fname()const=0;
  };

  enum IO {CLEAR, READ, WRITE, STREAM, FLUSH, COUNT};

  // CLEAR erases a file
  // READ reads the last element
  // WRITE writes a new element to the end
  // STREAM reads one element at a time from the beginning
  // FLUSH clears a buffer
  // COUNT counts the number of elements a file contains

  struct bad_io{
    IO io;
    bad_io(){}
    bad_io(IO io_param):io(io_param){}
  };

  void clear_file(const std::string &);
  void io_raw_data(const std::string &fname, double *obj, int lo, int hi,
		   IO io_prm, bool ask=true, bool endl=true);
  using BOOM::uint;
  //======================================================================
  class io_manager_base{
  protected:
    std::deque<Vec>  buf;
    string fname_;
    uint page_size;
  public:
    io_manager_base();
    io_manager_base(const string &Fname, uint p=1000);
    virtual ~io_manager_base();

    virtual void set_fname(const string &s);
    uint page()const;       // maximum number of items in buffer
    uint page(uint n);      // sets page size to n, returns previous
			    // page size
    uint buf_size()const;   // current number of items in buffer
    bool empty()const;      // whether buffer is empty
    virtual void push(const Vec &v)=0;// adds a parameter to the stack
    virtual Vec pop()=0;
    const string & file()const;
  };
  //======================================================================
  class output_manager : public io_manager_base{
  public:
    //    output_manager();
    output_manager(const string &Fname, uint p);
    ~output_manager();

    void clear_file()const;
    void write();           // appends prms to fname
    Vec pop();
    void push(const Vec &); // adds a parameter to the front
    virtual void set_fname(const string &s);
  };
  //======================================================================
  class input_manager : public io_manager_base{
    Vec prototype;
    std::streampos last_read_pos;
    bool first_time;
  public:
    //    input_manager(uint vec_sz);
    input_manager(uint vec_sz, const string &hist_file_name, uint p=1000);
    ~input_manager();

    void set_prototype(uint vec_size);
    uint vec_size()const;     // size of vectors held by input buffer
    void read();              // reads up to page_size entries from fname
    Vec read_last();  // gets last entry in file fname
    void reset_stream();
    void push(const Vec &);   // adds a parameter to the stack
    Vec pop();
    virtual void set_fname(const string &s);
  };

  //======================================================================
  class io_manager : public ParamIoManagerBase{
   public:
    io_manager();
    io_manager(const string &Fname, uint Bufsize=1000);
    virtual uint bufsize()const;
    virtual void set_bufsize(uint);
    virtual void reset_stream(uint dim);
    virtual void set_fname(const string &);
    virtual void output(const Vec &);        // reads Vec from buffer
    virtual void input(Vec &v, bool last_line=true);  // writes Vec to buffer
    virtual uint count_lines()const;
    virtual void clear_file();
    virtual void flush();
    virtual const string & fname()const{return fname_;}
   private:
    boost::scoped_ptr<output_manager> out;
    boost::scoped_ptr<input_manager> in;
    string fname_;
    uint bufsize_;
  };
  //======================================================================
  ostream & operator<<(ostream &, IO);
  istream & operator>>(istream &, IO);
  void intrusive_ptr_add_ref(io_manager_base *d);
  void intrusive_ptr_release(io_manager_base *d);
  void intrusive_ptr_add_ref(io_manager *d);
  void intrusive_ptr_release(io_manager *d);
  //----------------------------------------------------------------------


}
#endif // BOOM_IO_H
