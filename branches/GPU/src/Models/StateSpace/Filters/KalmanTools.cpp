/*
  Copyright (C) 2008 Steven L. Scott

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

#include <Models/StateSpace/Filters/KalmanTools.hpp>
#include <distributions.hpp>
namespace BOOM{

 double scalar_kalman_update(double y, Vec &a, Spd &P, Vec &K, double &F, double &v, bool missing,
                             const Vec & Z, double H, const Mat & T, Mat & L, const Spd & RQR){
       // y is y[t]

    // a starts as a[t] and ends as a[t+1]  a[t] = E(alpha[t]| Y^{t-1})
    // P starts as P[t] and ends as P[t+1]  P[t] = V(alpha[t]| Y^{t-1})
    // K is output as K[t]
    // Finv is output as Finv[t]
    // v is output as v[t]

    F = P.Mdist(Z) + H;
    double ans=0;
    if(!missing){
      K = T* (P * Z);
      K /= F;
      double mu = Z.dot(a);
      v = y-mu;
      ans =dnorm(y, mu, sqrt(F), true);

    }else{
      K = Z * 0;
      v = 0;
    }

    a = T * a;
    a += K * v;

    L = T.t();
    L.add_outer(Z, K, -1);  // L is the transpose of Durbin and Koopman's L
    P = T *P*L  + RQR;

    return ans;
 }

  void scalar_kalman_smoother_update(Vec &a, Spd &P, const Vec &K, double F, double v, const Vec & Z,
                                     const Mat &T, Vec & r, Mat &N, Mat & L){

    // updates a[t] and P[t] to condition on all Y
    // sets up r and N for use in the next recursion

    L = T.t();
    L.add_outer(Z,K, -1);   // L is the transpose of Durbin and Koopman's L
    r = L * r + Z*(v/F);
    N = sandwich(L,N);
    a += P*r;
    P -= sandwich(P,N);
  }

}
