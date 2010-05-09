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

#include <iostream>
#include <cmath>

#include <LinAlg/Matrix.hpp>
#include <LinAlg/Types.hpp>
#include <LinAlg/LinAlgExceptions.hpp>

#include <numopt.hpp>

namespace BOOM{
  using std::cerr;
  using std::endl;
  using LinAlg::matrix_not_positive_definite;
  /*----------------------------------------------------------------------
    functions for maximizing functions of several variables.
    ----------------------------------------------------------------------*/

  double max_nd0(Vec &x, Target tf){
    Negate f(tf);
    Vec wsp(x);
    int fc=0;
    double ans = nelder_mead_driver(x, wsp, f, 1e-8, 1e-8, 1.0, .5, 2.0,
 				    false, fc, 1000);
    return ans * -1;
  }
  //======================================================================
  double max_nd1(Vec &x, Target f, dTarget dtf, double eps){
    dNegate df(f, dtf);
    bool fail=false;
    int fcount=0;
    int gcount=0;
    int ntries=0;
    int maxntries= 5;
    double ans = bfgs(x,df, df, 200, eps, eps, fcount, gcount, fail);

    while(fail && ntries < maxntries){
      Vec g = x;
      nelder_mead_driver(x,g, Target(df), 1e-5, 1e-5, 1.0, .5, 2.0, false,
 			 fcount, 1000);
      fcount=gcount=0;
      fail=false;
      ans = bfgs(x,df, df, 200, 1e-8, 1e-8, fcount, gcount, fail);
      ++ntries;
    }
    return ans * -1;
  }

  //======================================================================
  double max_nd2(Vec &x, Vec &g, Mat &h, Target f, dTarget df, d2Target d2f,
 		 double leps){

    double ans=0;
    unsigned int ntries=0, maxtries = 5;
    d2Negate nd2f(f, df, d2f);
    /*------------ we should check that h is negative definite -----------*/
    int function_count = 0;
    bool happy = true;
    int gradient_count = 0;
    bool fail = false;
    do{
      try{
        ans=newton_raphson_min(x, g, h, nd2f, function_count, leps, happy);
        ++ntries;
        if(!happy){
          double bfgs_answer = bfgs(x, nd2f, nd2f, 200, leps, leps,
                                    function_count, gradient_count, fail);
          happy = (!fail) && (fabs(bfgs_answer - ans) < leps);
          // if bfgs thinks it succeeded and it got basically the same
          // answer as newton_raphson_min then we're done
        }
      }catch(matrix_not_positive_definite &e){
        bfgs(x, nd2f, nd2f, 200, leps, leps,
             function_count, gradient_count, fail);
        happy = false;
      }
    }while(!happy && ntries < maxtries);

    if(ntries>=maxtries){
      ostringstream err;
      err << "max_nd2 failed.   too many newton_raphson failures ";
      throw std::runtime_error(err.str());
    }

    /* othewise we're okay */
    g*= -1.0;
    h*= -1.0;
    return -1*ans;
  }
  /*======================================================================*/

}
