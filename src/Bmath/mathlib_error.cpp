#include "nmath.hpp"
#include <stdexcept>
#include <sstream>
#include <cpputil/ThrowException.hpp>
using namespace std;

namespace Rmath{
  void mathlib_error(const string &s){
    BOOM::throw_exception<std::runtime_error>(s); }

  void mathlib_error(const string &s, int d){
    ostringstream err;
    err << s << " " <<d << endl;
    BOOM::throw_exception<std::runtime_error>(err.str());
  }
  void mathlib_error(const string &s, double d){
    ostringstream err;
    err << s << " " <<d << endl;
    BOOM::throw_exception<std::runtime_error>(err.str());
  }
}
