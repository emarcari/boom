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
#include <Models/GaussianModelBase.hpp>
#include <distributions.hpp>


namespace BOOM{

  GaussianModelBase::GaussianModelBase()
    : DataPolicy(new GaussianSuf())
  {}

  GaussianModelBase::GaussianModelBase(const std::vector<double> &y)
      : DataPolicy(new GaussianSuf())
  {
    DataPolicy::set_data_raw(y.begin(), y.end());
  }

  double GaussianModelBase::Logp(double x, double &g, double &h, uint nd)const{
    double m = mu();
    double ans = dnorm(x, m, sigma(), 1);
    if(nd>0) g = -(x-m)/sigsq();
    if(nd>1) h = -1.0/sigsq();
    return ans;
  }

  double GaussianModelBase::Logp(const Vec &x, Vec &g, Mat &h, uint nd)const{
    double X=x[0];
    double G(0),H(0);
    double ans = Logp(X,G,H,nd);
    if(nd>0) g[0]=G;
    if(nd>1) h(0,0)=H;
    return ans;
  }

  double GaussianModelBase::ybar()const{return suf()->ybar();}
  double GaussianModelBase::sample_var()const{return suf()->sample_var();}


  void GaussianModelBase::add_mixture_data(Ptr<Data> dp, double prob){
    double y = DAT(dp)->value();
    suf()->add_mixture_data(y, prob);
  }

  double GaussianModelBase::sim()const{ return rnorm(mu(), sigma()); }

  void GaussianModelBase::add_data_raw(double x){
    NEW(DoubleData, dp)(x);
    this->add_data(dp);
  }



}
