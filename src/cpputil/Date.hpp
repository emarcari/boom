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

#ifndef BOOM_DATE_HPP
#define BOOM_DATE_HPP

#include <string>
#include <BOOM.hpp>

namespace BOOM{
  struct bad_date{};

  enum month_names{unknown_month=0, Jan=1, Feb, Mar, Apr, May, Jun, Jul,
 		   Aug, Sep, Oct, Nov, Dec};

  enum day_names{Sat=0, Sun, Mon, Tue, Wed, Thu, Fri};
  enum calendar_format{Full,full,Abbreviations, abbreviations,numeric};

  inline unsigned short days_in_month(month_names month, bool leap_year=false){
    static unsigned short ndays[]={0,31,28,31,30,31,30,31,31,30,31,30,31};
    if(month==Feb) return leap_year ? 29 : 28;
    else return ndays[month];}

  inline bool is_leap_year(int yyyy){
    //divisible by 4, not if divisible by 100, but true if divisible by 400
    return (!(yyyy % 4))  && ((yyyy % 100) || (!(yyyy % 400)));
  }

  struct unknown_month_name{
    string s;
    unknown_month_name(const string &m) : s(m){}
  };

  month_names str2month(const string &m);

  ostream & operator<<(ostream &, const day_names &);

  class Date{
  public:
    enum print_order{mdy, dmy, ymd};
    enum date_format{slashes, dashes, script};
  private:
    month_names m_;
    int d_;
    int y_;
    void check();
    static Date day_zero;  // default is Jan 1 2000
    static calendar_format month_format;
    static calendar_format day_format;
    static date_format df;
    static print_order po;
    Date & start_next_month();
    Date & end_prev_month();
  public:
    Date();
    Date(int julian_date);
    Date(int m, int dd, int yyyy);
    Date(month_names m, int dd, int yyyy);
    Date(const string &mdy, char delim='/');
    Date(const string &m, int d, int yyyy);
    Date(const Date & rhs);

    Date & operator=(const Date &rhs);

    unsigned short days_left_in_month()const;//jan 31 = 0, jan1=30
    unsigned short days_into_year()const;    // jan 1 is 1
    unsigned short days_left_in_year()const; // dec 31 = 0; jan 1 = 364
    bool is_leap_year()const;

    Date & operator++();   // next day
    Date operator++(int);   // next day (postfix)
    Date & operator--();   // previous day
    Date operator--(int); // previous day (postfix);
    Date & operator+=(int n);  // add n days
    Date & operator-=(int n);  // subtract n days

    Date operator+(int n);
    Date operator-(int n);

    bool operator==(const Date &rhs)const;  // comparison operators
    bool operator!=(const Date &rhs)const;
    bool operator<(const Date &rhs)const;
    bool operator>(const Date &rhs)const;
    bool operator<=(const Date &rhs)const;
    bool operator>=(const Date &rhs)const;

    month_names month()const;
    unsigned short day()const;
    day_names day_of_week()const;
    int year()const;
    long julian()const;    // zero Date is Jan 1 2000

    static void set_zero(const Date &d);
    static void set_zero(int m, int d, int y);

    static void solve_for_zero(int m, int d, int y, int jdate);
    static void set_month_format(calendar_format f);
    static void set_day_format(calendar_format f);
    static void set_print_order(print_order d);
    static void set_date_format(date_format f);

    std::ostream & display(std::ostream &)const;
    std::ostream & display_month(std::ostream &)const;
    string str()const;
  };

  Date mdy2Date(int m, int d, int yyyy);
  Date dmy2Date(int d, int m, int yyyy);
  Date ymd2Date(int yyyy, int m, int d);

  Date mdy2Date(const string &, char delim='/');
  Date dmy2Date(const string &, char delim='/');
  Date ymd2Date(const string &, char delim='/');

  int operator-(const Date &d1, const Date &d2);

  std::ostream & operator<<(std::ostream &, const Date &d);
  std::ostream & display(std::ostream &, day_names,
 			 calendar_format=Abbreviations);

  Date guess_date_format(const string &s, char delim='/');
  //============================================================
  class timepoint{
    Date dt_;  // date today
    double t_; // fractional day \in [0,1)
    double hms2t(const string &hms, char delim=':')const;
    static char hms_sep_;
    static char date_time_sep_;
    static const double seconds_in_day;
    static const double minutes_in_day;
    static const double hours_in_day;
    static const double hour_size;
    static const double minute_size;
    static const double second_size;
  public:
    timepoint();
    timepoint(const Date &d, const string &hms, char delim=':');
    timepoint(double  julian_time);
    timepoint(const string &mdy, const string &hms);
    timepoint(const timepoint & rhs);

    timepoint & operator=(const timepoint & rhs);

    bool operator==(const timepoint &rhs)const;
    bool operator<(const timepoint &rhs)const;
    bool operator!=(const timepoint &rhs)const;
    bool operator>=(const timepoint &rhs)const;
    bool operator<=(const timepoint &rhs)const;
    bool operator>(const timepoint &rhs)const;

    double time_left_in_day()const;
    unsigned short hour()const;    // 0..23
    unsigned short minute()const;  // 0..59
    unsigned short second()const;  // 0..59
    unsigned short hours_left_in_day()const;  // 0..23
    unsigned short minutes_left_in_hour()const; // 0..59
    unsigned short seconds_left_in_minute()const; // 0..59

    timepoint & operator+=(double ndays);
    timepoint & operator-=(double ndays);
    timepoint  operator+(double ndays);
    timepoint  operator-(double ndays);

    Date date()const;
    static void hms_sep(char c);
    static void date_time_sep(char c);

    friend ostream & operator<<(ostream &, const timepoint &tm);
  };
  double operator-(const timepoint &t1, const timepoint &t2);
  ostream & operator<<(ostream &, const timepoint &tm );
}

#endif // BOOM_DATE_HPP
