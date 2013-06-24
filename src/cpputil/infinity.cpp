#include "math_utils.hpp"
#include <limits>
#include <cmath>

namespace BOOM{

  double infinity(){
    return std::numeric_limits<double>::infinity(); }

  double negative_infinity(){
    return -1*std::numeric_limits<double>::infinity(); }

  bool finite(double x){
    return std::isfinite(x);}

//   bool finite(double x){
//     return x > negative_infinity()
//       && x < infinity()
//       && !isnan(x);
//   }

}
