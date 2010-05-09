#include "nmath.hpp"
#include <stdexcept>
#include <sstream>

using namespace std;

namespace Rmath{
  void mathlib_error(const string &s){
    throw std::runtime_error(s); }

  void mathlib_error(const string &s, int d){
    ostringstream err;
    err << s << " " <<d << endl;
    throw std::runtime_error(err.str());
  }
  void mathlib_error(const string &s, double d){
    ostringstream err;
    err << s << " " <<d << endl;
    throw std::runtime_error(err.str());
  }
}
