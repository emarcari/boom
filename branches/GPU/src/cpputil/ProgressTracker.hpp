#ifndef BOOM_PROGRESS_TRACKER_CLASS_HPP
#define BOOM_PROGRESS_TRACKER_CLASS_HPP

#include <BOOM.hpp>
#include <cpputil/Ptr.hpp>
#include <cpputil/RefCounted.hpp>

namespace BOOM{
  class ProgressTracker:  private RefCounted{
    string fname;
    ostream * msg_;
    uint nskip;
    uint n;
    string sep;
    bool owns_msg;
    void start(const string & prog_name);
    ProgressTracker(const ProgressTracker &) : RefCounted(){}
  public:
    ProgressTracker(const string &dname, uint nskip=100,
                    bool restart=false, const string & prog_name="",
                    bool keep_existing_msg=false);
    ProgressTracker(uint nskip=100, const string & prog_name="");
    ProgressTracker(ostream &out, uint nskip=100, const string & prog_name="");
    ~ProgressTracker();
    ProgressTracker & operator++(){update(); return *this;}
    ProgressTracker & operator++(int){update(); return *this;}
    void update();
    uint restart();
    void set_niter(uint n);

    ostream & msg();

    friend void intrusive_ptr_add_ref(ProgressTracker *m);
    friend void intrusive_ptr_release(ProgressTracker *m);
  };
  void intrusive_ptr_add_ref(ProgressTracker *m);
  void intrusive_ptr_release(ProgressTracker *m);
}
#endif// BOOM_PROGRESS_TRACKER_CLASS_HPP
