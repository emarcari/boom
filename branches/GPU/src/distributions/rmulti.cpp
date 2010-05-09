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

#include <cmath>
#include <distributions.hpp>
#include <LinAlg/Types.hpp>
#include <LinAlg/Vector.hpp>

#include <stdexcept>
#include <sstream>

using namespace std;
namespace BOOM{

  int rmulti(int lo, int hi){
    return rmulti_mt(GlobalRng::rng, lo, hi);}

  int rmulti_mt(RNG & rng, int lo, int hi){
    // draw a random integer between lo and hi with equal probability
    double tmp = runif_mt(rng, lo+0.0,hi+1.0);
    return (int)floor(tmp);
  }
  uint rmulti(const Vec &prob){
    return rmulti_mt(GlobalRng::rng, prob);}

  uint rmulti_mt(RNG & rng, const Vec &prob){

    /* This function draws a deviate from the multiBernoulli
       distribution with states from lo to hi.  The probability
       vector need not sum to 1, it only needs to be specified up to a
       proportionality constant */

    double probsum = prob.abs_norm();
    if(!finite(probsum)){
      std::ostringstream err;
      err << "infinite or NA probabilities supplied to rmulti:  prob = " << prob << endl;
      throw std::runtime_error(err.str());
    }
    if(probsum<=0){
      std::ostringstream err;
      err << "zero or negative normalizing constant in rmulti:  prob = " << prob << endl;
      throw std::runtime_error(err.str());
    }
    double tmp=runif_mt(rng, 0,probsum);

    double psum=0;
    uint n = prob.size();
    for(uint i = 0; i<n; ++i){
      psum+=prob(i);
      if(tmp<=psum) return i;}
    ostringstream msg;
    msg << "rmulti failed:  prob = " << prob << endl
	<< "psum = " << psum << endl;
    throw std::runtime_error(msg.str());
    return 0;
  }
}
