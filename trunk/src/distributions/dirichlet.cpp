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

#include <distributions.hpp>
#include <LinAlg/Types.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <cpputil/math_utils.hpp>

#include <stdexcept>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>

namespace BOOM{
  using std::string;
  using std::ostringstream;
  using std::endl;

  inline void illegal_parameter_value(const Vec &n, const string &fname,
				      const string & prm_name){
    ostringstream msg;
    msg << "illegal_parameter_value in " << fname << endl
	<< prm_name << " = " << n << endl;
    throw std::runtime_error(msg.str());
  }

  //======================================================================
  template <class V>
  bool dirichlet_exception(const V & nu, uint j){
    ostringstream err;
    err <<  "element " << j << " was zero in rdirichlet, with nu = "
        << nu << endl;
    throw std::runtime_error(err.str());
    return false;
  }

  template <class V>
  Vec rdirichlet_impl(RNG & rng, const V & nu){
    uint N = nu.size();
    Vec x(N);
    if(N==0) return x;
    if(N==1){
      x=1.0;
      return x;
    }

    double sum=0;
    for(uint j=0; j<N; ++j){
      double n = nu(j);
      if(n <=0 ) illegal_parameter_value(nu, "rdirichlet", "nu");
      x(j)=rgamma_mt(rng, n, 1);
      assert(x(j)>0 || dirichlet_exception(nu, j) );
      sum+=x(j); }
    if(!std::isnormal(sum)){
      ostringstream err;
      err << "infinite, NaN, or denormalized sum in rdirichlet_impl.  sum = " << sum << endl
          << "x = " << x << endl
          << "nu = " << nu << endl;
      throw std::runtime_error(err.str());
    }
    if(sum<=0){
      ostringstream err;
      err << "non-positive sum in rdirichlet_impl.  sum = " << sum << endl
          << "x = " << x << endl
          << "nu = " << nu << endl;
      std::runtime_error(err.str());
    }
    x/=sum;
    return x;
  }
  Vec rdirichlet_mt(RNG & rng, const Vec & nu){
    return rdirichlet_impl(rng, nu);}
  //======================================================================
  Vec rdirichlet(const Vec & nu){
    return rdirichlet_impl(GlobalRng::rng, nu); }
  //======================================================================
  Vec rdirichlet_mt(RNG & rng, const VectorView & nu){
    return rdirichlet_impl(rng, nu);}
  //======================================================================
  Vec rdirichlet(const VectorView & nu){
    return rdirichlet_impl(GlobalRng::rng, nu); }
  //======================================================================
  Vec rdirichlet_mt(RNG & rng, const ConstVectorView & nu){
    return rdirichlet_impl(rng, nu);}
  //======================================================================
  Vec rdirichlet(const ConstVectorView & nu){
    return rdirichlet_impl(GlobalRng::rng, nu); }
  //======================================================================

  template <class V1, class V2>
  double ddirichlet_impl(const V1 & x, const V2 &nu, bool logscale){
    /* x(lo..hi) is a dirichlet RV (non-negative and sums to 1).
       nu(lo..hi) is the parameter (non-negative).  */

    double ans(0), sum(0), xsum(0);
    for(uint i=0; i<x.size(); ++i){
      double xi= x(i);
      if(xi>1 || xi<0) return logscale ? BOOM::infinity(-1) : 0;
      xsum+= xi;

      double nui = nu(i);
      sum+=nui;
      ans+= (nui-1.0)*log(xi) - lgamma(nui);
    }
    const double eps = 1e-5;  // std::numeric_limits<double>::epsilon()
    if( fabs(xsum-1.0) >  eps ){
      return logscale ? BOOM::infinity(-1) : 0;
    }
    ans+= lgamma(sum);
    return logscale? ans: exp(ans);
  }

  double ddirichlet(const Vec & x, const Vec & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}
  double ddirichlet(const VectorView & x, const Vec & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}
  double ddirichlet(const Vec & x, const VectorView & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}
  double ddirichlet(const VectorView & x, const VectorView & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}

  double ddirichlet(const ConstVectorView & x, const Vec & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}
  double ddirichlet(const Vec & x, const ConstVectorView & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}
  double ddirichlet(const ConstVectorView & x, const ConstVectorView & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}
  double ddirichlet(const ConstVectorView & x, const VectorView & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}
  double ddirichlet(const VectorView & x, const ConstVectorView & nu, bool logscale){
    return ddirichlet_impl(x,nu,logscale);}



  //======================================================================
  Vec mdirichlet( const Vec &nu){
    /* returns x(lo..hi): mode of the dirichlet distribution with
       parameter nu(lo..hi) each nu(i)>0.  */

    double nc=0;
    uint n = nu.size();
    Vec x(nu-1.0);
    for(uint i=0; i<n; ++i){
      if(x[i]<0) x[i]=0;
      nc+=x[i];
    }
    x/=nc;
    return x;
  }

  double dirichlet_loglike(const Vec &nu, Vec *g, Mat *h,
			   const Vec & sumlogpi, double nobs){

    uint n = nu.size();

    double sum=0;
    bool flag=false;
    for(uint i=0; i<n; ++i){  /* check for illegal parameter values */
      sum+=nu(i);
      if(nu(i)<=0){
 	flag=true;
 	break;}}
    if(flag){
      for(uint i=0; i<n; ++i){
 	if(g){
 	  (*g)(i)= -nu(i);
 	  if(h) for(uint j=0; j<n; ++j) (*h)(i,j)= (i==j)?1:0;}}
      return BOOM::infinity(-1);}

    double ans= nobs*lgamma(sum);
    double tmp=0.0, tmp1=0.0;
    if(g) tmp= nobs*digamma(sum);
    if(h) tmp1=nobs*trigamma(sum);

    for(uint i=0; i<n; ++i){
      ans+= (nu(i)-1)*sumlogpi(i)-nobs*lgamma(nu(i));
      if(g){
 	(*g)(i)= tmp + sumlogpi(i)-nobs*digamma(nu(i));
 	if(h){
 	  for(uint j=0; j<n; ++j){
 	    (*h)(i,j) =tmp1- (i==j? nobs*trigamma(nu(i)):0); }}}}

    return ans;


  }

}