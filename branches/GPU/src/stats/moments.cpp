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

#include "moments.hpp"
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>

namespace BOOM{
  inline double SQ(double x){return x*x;}
  Vec mean(const Mat &m){
    int nr = nrow(m);
    Vec ave(nr, 1.0/nr);
    Vec ans = ave * m;
    return ans;
  }

  Spd var(const Mat &m){
    Spd ans(m.ncol(), 0.0);
    Vec mu = mean(m);
    for(uint i = 0; i<m.nrow(); ++i){
      Vec tmp = m.row(i)- mu;
      ans.add_outer(tmp);}
    ans/=(m.nrow()-1);
    return ans;
  }

  Spd cor(const Mat &m){
    Spd V = var(m);
    Vec sd = sqrt(diag(V));
    Spd d(sd.size());
    d.set_diag(1.0/sd);

    Spd ans = d * V * d;
    return ans;
  }

  double mean(const Vec &x){ return x.sum()/x.size();}
  double var(const Vec &x){
    double mu = mean(x);
    double sumsq = 0;
    uint n = x.size();
    for(uint i =0 ; i<n; ++i) sumsq+= SQ(x[i]-mu);
    return sumsq/(n-1);
  }

  double mean(const std::vector<double> &x){
    if(x.size()==0) return 0.0;
    double ans = 0;
    for(uint i=0; i<x.size(); ++i) ans+= x[i];
    return ans/x.size();
  }

  double var(const std::vector<double> &x){
    if(x.size()<=1) return 0.0;
    double ans = 0;
    double mu = mean(x);
    for(uint i=0; i<x.size(); ++i) ans+= SQ(x[i] - mu);
    return ans/(x.size()-1);
  }

}
