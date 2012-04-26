#include "file_utils.hpp"
#include <fstream>
#include <cpputil/ThrowException.hpp>
using namespace std;
namespace BOOM{
  uint count_lines(const string &fname){
    ifstream in(fname.c_str());
    if(!in){
      string err = "couldn't find file named " + fname;
      throw_exception<runtime_error>(err.c_str());
    }

    uint cnt=0;
    while(in){
      string line;
      getline(in, line);
      if(!in) break;
      ++cnt;
    }
    return cnt;
  }
}
