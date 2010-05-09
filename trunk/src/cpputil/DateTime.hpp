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

#ifndef BOOM_DATE_TIME_HPP
#define BOOM_DATE_TIME_HPP
#include <cpputil/Date.hpp>
#include <boost/operators.hpp>

namespace BOOM{

class DateTime;
double operator-(const DateTime &, const DateTime &);

class DateTime
    : public boost::totally_ordered<DateTime>,
      public boost::additive<DateTime,double>
{
 public:
  DateTime();
  DateTime(const Date &, double fraction_of_day);
  DateTime(const Date &, uint hour, uint min, double sec);
  double seconds_to_next_day()const;
  double seconds_into_day()const;

  bool operator<(const DateTime &rhs)const;
  bool operator==(const DateTime &rhs)const;


  DateTime & operator+=(double days);
  DateTime & operator-=(double days);

  long hour()const;    // 0..23
  long minute()const;  // 0..59
  long second()const;  // 0..59

  unsigned short hours_left_in_day()const;  // 0..23
  unsigned short minutes_left_in_hour()const; // 0..59
  unsigned short seconds_left_in_minute()const; // 0..59

  ostream & print(ostream &)const;

  // convert between time scales.  Output is the relevant fraction of a day.
  // example:  hours(1) = 1.0/24, weeks(2) = 14
  static double weeks(double duration);
  static double days(double duration);
  static double hours(double duration);
  static double minutes(double duration);
  static double seconds(double duration);
  static double milliseconds(double duration);
  static double microseconds(double duration);

 private:
  Date d_;
  double t_;  // fraction of day [0,1)
  static const uint seconds_in_day_;
  static const uint minutes_in_day_;
  static const uint hours_in_day_;
  static const double milliseconds_in_day_;
  static const double microseconds_in_day_;
  friend double operator-(const DateTime &, const DateTime &);
};


ostream & operator<<(ostream &out, const DateTime & dt);


}
#endif// BOOM_DATE_TIME_HPP
