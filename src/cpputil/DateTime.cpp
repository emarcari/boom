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
#include "DateTime.hpp"
#include <cmath>
#include <cassert>

namespace BOOM{
typedef DateTime DT;

const uint DT::seconds_in_day_(86400);
const uint DT::minutes_in_day_(1440);
const uint DT::hours_in_day_(24);
const double DT::milliseconds_in_day_(seconds_in_day_ * 1000.0);
const double DT::microseconds_in_day_(seconds_in_day_ * 1.0e+6);

DT::DateTime()
    : t_(0.0)
{
  assert(t_ >=0 && t_<=1.0);
}

DT::DateTime(const Date &d, double fraction_of_day_)
    : d_(d),
      t_(fraction_of_day_)
{
  assert(t_ >=0 && t_<=1.0);
}

DT::DateTime(const Date &d, uint hour, uint min, double sec)
    : d_(d)
{
  assert(hour<24);
  assert(min<60);
  assert(sec <= 60.0000);
  t_ = hour/24.0 + min/24.0/60 + sec/24/3600;
  assert(t_>=0 && t_<=1.0);
}

double DT::seconds_to_next_day()const{ return seconds_in_day_*(1 - t_) ;}
double DT::seconds_into_day()const{ return seconds_in_day_ * t_;}

bool DT::operator<(const DT &rhs)const{
  if(d_ < rhs.d_) return true;
  if(d_ > rhs.d_) return false;
  if(t_ < rhs.t_) return true;
  return false;
}

bool DT::operator==(const DT & rhs)const{
  if(d_!=rhs.d_) return false;
  if( t_ < rhs.t_ || t_>rhs.t_) return false;
  return true;
}

inline double rem(double x, double y){
  double v = floor(x/y);
  return x-v*y;
}



DT & DT::operator+=(double days){
  if(days<0) return (*this)-=(-days);

  t_ += days;
  if(t_>=1){
    double frac = rem(t_,1.0);
    long ndays = lround(t_ - frac);
    d_ += ndays;
    t_ = frac;
  }

  return *this;
}

DT & DT::operator-=(double days){
  if(days<0) return (*this)+=(-days);

  t_ -= days;
  if(t_<0){
    double frac = rem(t_,1.0); // a negative number in (-1,0]
    long ndays = lround(floor(t_));  // a negative number <= t_
    d_ += ndays;
    t_ = 1-frac;
  }
  return *this;
}

long DT::hour()const{
  return lround(floor(t_ * hours_in_day_)); }

long DT::minute()const{
  double m = rem(t_ * minutes_in_day_, 60);
  assert(m>=0);
  return lround(floor(m));
}

long DT::second()const{
  double s = rem(t_ * seconds_in_day_, 60);
  assert(s>=0);
  return lround(floor(s));}

ostream & DT::print(ostream &out)const{
  double hr = hour();
  double min = minute();
  double sec = second();
  double frac = t_ - (hr/24 + min/24/60 + sec/24/60/60);
  frac*=seconds_in_day_;
  sec += frac;
  out << d_ << " " <<hr <<":" <<min<<":" <<sec;
  return out;
}

ostream & operator<<(ostream &out, const DT & dt){
  return dt.print(out);}

double operator-(const DateTime &t1, const DateTime &t2){
  int ndays = t1.d_ - t2.d_;
  double dt = t1.t_ - t2.t_;
  return dt + ndays;
}


double DT::weeks(double t){return t*7;}
double DT::days(double t){return t;}
double DT::hours(double t){return t/hours_in_day_;}
double DT::minutes(double t){return t/minutes_in_day_;}
double DT::seconds(double t){return t/seconds_in_day_;}
double DT::milliseconds(double t){return t/milliseconds_in_day_;}
double DT::microseconds(double t){return t/microseconds_in_day_;}


}
