#include "ProgressTracker.hpp"
#include <cpputil/file_utils.hpp>
#include <cpputil/date_utils.hpp>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

namespace BOOM{

  ProgressTracker::ProgressTracker(const string &dname,
                                   uint Nskip,
                                   bool restart,
                                   const string &prog_name, 
                                   bool keep_existing_msg)
    : fname(add_to_path(dname, "msg")),
      nskip(Nskip),
      n(0),
      sep(" =-=-=-=-=-=-=-=-= "),
      owns_msg(true)
  {
    if(restart || keep_existing_msg) 
      msg_ = new ofstream(fname.c_str(), std::ios::app);
    else msg_ = new ofstream(fname.c_str(), std::ios::trunc);
    start(prog_name);
    if(restart) n = this->restart();
  }

  ProgressTracker::ProgressTracker(uint Nskip, const string &prog_name)
    : nskip(Nskip),
      n(0),
      sep(" =-=-=-=-=-=-=-=-= "),
      owns_msg(false)
  {
    msg_ = &std::cout;
    start(prog_name);
  }

  ProgressTracker::ProgressTracker(ostream &out, uint Nskip,
				     const string & prog_name)
    : fname(""),
      msg_(&out),
      nskip(Nskip),
      n(0),
      sep(" =-=-=-=-=-=-=-=-= "),
      owns_msg(false)
  {
    start(prog_name);
  }


  ProgressTracker::~ProgressTracker(){
    msg() << sep << " All finished " << get_date() << sep << endl;
    if(owns_msg) delete msg_;
  }

  void ProgressTracker::start(const string & prog_name){
    string space = " ";
    uint n = prog_name.size();
    if(n>0 && prog_name[n-1]==' ') space = "";
    msg() << sep << "Starting program: " << prog_name  << space << get_date()
	 << sep << endl << endl;
  }

  void ProgressTracker::set_niter(uint niter){ n = niter;}

  void ProgressTracker::update(){
    ++n;
    if(n==1 || n%nskip ==0){
      char stamp[100];
      string now = get_date(stamp);
      const string sep = " =-=-=-=-=-=-=-=-= ";
      msg() << sep << "Iteration " << n << " " << now << sep <<endl;
    }
  }
  //======================================================================
  unsigned int ProgressTracker::restart(){
    if(fname.size()==0){
      ostringstream out;
      out << "cannot use ProgressTracker to restart without first "
	  << "setting history directory";
      throw std::logic_error(out.str());
    }
    ifstream in(fname.c_str());
    gll(in);   // put msg to start of final line
    string s;
    in >> s >> s >> n;  // reads separator, Iteration, and it number
    return n;
  }
  //======================================================================
  ostream & ProgressTracker::msg(){return *msg_;}
  //======================================================================
  void intrusive_ptr_add_ref(ProgressTracker *s){
    s->up_count();}
  void intrusive_ptr_release(ProgressTracker *s){
    s->down_count();
    if(s->ref_count()==0) delete s; }

}
