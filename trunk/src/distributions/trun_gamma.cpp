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

#include <cmath>                          // log
#include <distributions.hpp>              // rgamma, runif
#include <cpputil/math_utils.hpp>         // infinity
#include <distributions/trun_gamma.hpp>

namespace BOOM{

  double rtg_init(double x, double a, double b, double cut, double logpstar);
  double rtg_slice(RNG & , double x,double a,double b,double cut);

  double dtrun_gamma(double x, double a, double b, double cut, bool logscale){
    /*
     * return the un-normalized density of a gamma(a,b) random
     * variable x, given x > cut
     */

    if(a < 0 || b< 0 || cut < 0 || x < cut) return BOOM::infinity(-1);

    double ans = (a-1)*log(x) - b * x;
    return logscale ? ans : exp(ans);
  }

  //----------------------------------------------------------------------

  double rtrun_gamma(double a,double b,double cut, unsigned n){
    return rtrun_gamma_mt(GlobalRng::rng, a, b, cut, n);}

  double rtrun_gamma_mt(RNG & rng, double a,double b,double cut, unsigned n){
    double mode = (a-1)/b;
    double x = 0;
    if(cut < mode){    // rejection sampling
      while(x<cut) x = rgamma_mt(rng, a,b);
      return x;
    }
    x = cut;
    for(unsigned i=0; i<n; ++i) x = rtg_slice(rng, x,a,b,cut);
    return x;
  }

  //----------------------------------------------------------------------

  double rtg_init(double x, double a, double b, double cut, double logpstar){
    /*
     * finds a value of x such that dtrun_gamma(x,a,b,cut,true) <
     * logpstar.  This function will only be called if cut > mode, in
     * which case dtrun_gamma is a decreasing function.
     */

    double f = dtrun_gamma(x,a,b,cut,true) - logpstar;
    double fprime = ((a-1)/x)  - b;
    while(f>0){
      x -= f/fprime;
      f = dtrun_gamma(x,a,b,cut,true) - logpstar;
      fprime = ((a-1)/cut)  - b;
    }
    return x;
  }
  //----------------------------------------------------------------------
  double rtg_slice(RNG &rng, double x,double a,double b,double cut){
    double logpstar = dtrun_gamma(x,a,b,cut,true) - rexp(1);
    double lo = cut;
    double hi = rtg_init(x,a,b,cut, logpstar);
    x = runif_mt(rng, lo,hi);
    while( dtrun_gamma(x,a,b, cut, true) < logpstar){
      hi = x;
      x = runif_mt(rng, lo,hi);
    }
    return(x);
  }


}
