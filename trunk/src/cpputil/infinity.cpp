#include "math_utils.hpp"
#include <limits>
#include <cmath>

namespace BOOM{

  double infinity(int sgn){
    return (sgn>0 ? 1:-1)*std::numeric_limits<double>::infinity(); }

  bool finite(double x){
    return std::isfinite(x);}

//   bool finite(double x){
//     return x > infinity(-1)
//       && x < infinity(1)
//       && !isnan(x);
//   }

}
