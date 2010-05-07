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

#include <cpputil/Date.hpp>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <cpputil/string_utils.hpp>
#include <cassert>
#include <ctime>
namespace BOOM{

  ostream & operator<<(ostream & out, const day_names &d){
    if(d==Sat) out << "Saturday";
    else if(d==Sun) out << "Sunday";
    else if(d==Mon) out << "Monday";
    else if(d==Tue) out << "Tuesday";
    else if(d==Wed) out << "Wednesday";
    else if(d==Thu) out << "Thursday";
    else if(d==Fri) out << "Friday";
    else{
      throw std::runtime_error("Unknown day name");
    }
    return out;
  }

  Date::Date() : m_(unknown_month), d_(0), y_(0){
    time_t t=time(0);
    tm * now = localtime(&t);
    m_ = month_names(now->tm_mon + 1);
    d_ = now->tm_mday;
    y_ = now->tm_year + 1900;
  }

  Date::Date(int m, int dd, int yyyy)
    : m_(month_names(m)), d_(dd), y_(yyyy){
    check(); }

  Date::Date(month_names m, int dd, int yyyy)
    : m_(m), d_(dd), y_(yyyy){
    check(); }

  Date::Date(const string &m, int d, int yyyy)
    : m_(str2month(m)), d_(d), y_(yyyy){
    check();}

  Date::Date(const string &mdy, char delim){
    std::vector<string> tmp = split_delimited(mdy, delim);
    m_ = str2month(tmp[0]) ;
    istringstream dstr(tmp[1]);
    dstr >> d_;
    istringstream ystr(tmp[2]);
    ystr >> y_;
    check();
  }

  Date::Date(int n){
    operator=(day_zero);
    operator+=(n);
  }

  Date::Date(const Date &rhs) : m_(rhs.m_), d_(rhs.d_), y_(rhs.y_){}


  void Date::check(){
    if(d_ < 1 || d_ > days_in_month(m_, is_leap_year() ))
      throw bad_date();
  }

  inline unsigned short days_before_first(month_names m, bool leap){
    // returns number of days in the year before first the first of
    // each month
    static unsigned short tab[]={0,0,31, 59, 90, 120, 151, 181, 212,
 				 243, 273, 304, 334};
    unsigned short ans = tab[m];
    if(leap && m>Feb) ++ans;
    return ans;
  }

  inline unsigned int num_leap_years(int y1, int base_year){
    // should return number of complete leap years between
    // January 1 of base_year and a date interior to y1
    unsigned int ans=0;
    if(y1 > base_year){
      while(base_year< y1 && !is_leap_year(base_year)) ++base_year;
      if(base_year==y1) return 0;
      ans= (y1-base_year)/4 + (is_leap_year(base_year)) - is_leap_year(y1);
    }else if(y1< base_year){
      do{
 	++y1;
      }while(y1<base_year && !is_leap_year(y1));
      if(y1==base_year) return 0;
      ans= (base_year-y1)/4 + is_leap_year(y1) - is_leap_year(base_year);
    }
    return ans;
  }

  bool Date::is_leap_year()const{return BOOM::is_leap_year(y_);}

  Date & Date::operator=(const Date &rhs){
    if(&rhs==this) return *this;
    y_ = rhs.y_;
    m_ = rhs.m_;
    d_ = rhs.d_;
    return *this;
  }

  Date & Date::operator++(){
    if(days_left_in_month()==0) start_next_month();
    else ++d_;
    return *this; }

  Date & Date::operator--(){
    if(d_==1) end_prev_month();
    else --d_;
    return *this;}

  Date Date::operator++(int){
    Date tmp(*this);
    operator++();
    return tmp; }

  Date Date::operator--(int){
    Date tmp(*this);
    operator--();
    return tmp;
  }

  unsigned short Date::days_left_in_month()const{
    return days_in_month(m_, is_leap_year()) - d_; }

  unsigned short Date::days_into_year()const{
    return days_before_first(m_, is_leap_year()) + d_;  }

  unsigned short Date::days_left_in_year()const{
    bool leap = is_leap_year();
    return 365+leap - days_into_year();  }

  month_names Date::month()const{return m_;}
  unsigned short Date::day()const{return d_;}
  int Date::year()const{return y_;}

  day_names Date::day_of_week()const{
    // Jan 1 2000 was a saturday
    long nd = julian() - Date(Jan,1,2000).julian();
    if(nd>0) return day_names(nd%7);
    if(nd==0) return Sat;
    return day_names(((nd%7)+7)%7);
  }

  long Date::julian()const{
    int dy = y_-day_zero.year();
    int nleap= num_leap_years(y_, day_zero.year());
    long ans=0;
    if(dy>=0) ans = 365*dy + nleap + days_into_year();
    else ans = 365*dy -nleap + days_into_year();
    return ans-1;
  }

  bool Date::operator==(const Date &rhs)const{
    if(y_==rhs.y_ && m_ == rhs.m_ && d_ == rhs.d_) return true;
    return false;}
  bool Date::operator<(const Date &rhs)const{
    if(*this==rhs) return false;
    if(year()<rhs.year()){
      return true;
      if(year()>rhs.year()) return false;
    }
    // same year
    if(month() < rhs.month()){
      return true;
      if(month() > rhs.month()) return false;
    }
    // same month
    return(day() < rhs.day());
  }
  bool Date::operator<=(const Date &rhs)const{
    if(*this==rhs) return true;
    if(*this<rhs) return true;
    return false;  }
  bool Date::operator!=(const Date &rhs)const{return !((*this)==rhs);}
  bool Date::operator>(const Date &rhs)const{ return !(*this<=rhs);}
  bool Date::operator>=(const Date &rhs)const{return !(*this<rhs);}


  Date & Date::start_next_month(){
    d_ = 1;
    if(m_==Dec){
      ++y_;
      m_ = Jan;
    }else m_ = month_names(m_+1);
    return *this;
  }

  Date & Date::end_prev_month(){
    if(m_==Jan){
      d_ = 31;
      m_ = Dec;
      --y_;
    }else{
      m_ = month_names(m_-1);
      d_ = days_in_month(m_, is_leap_year());
    }
    return *this;
  }


  Date & Date::operator+=(int n){
    if(n==0) return *this;
    if(n<0) return operator-=(-n);

    if(n>=days_left_in_year()){
      n-= days_left_in_year();
      m_=Jan;
      d_=1;
      ++y_;

      if(n>1461 ){       // number of days in 4 years
 	int dy4 = n/1461;
 	y_+= 4*dy4;
 	n = n%1461;
      }
      while(n>365){
 	n-= 365 + is_leap_year();
 	++y_;}}
    // now n<=365;

    while(n > days_left_in_month()){
      n-= days_left_in_month()+1;
      start_next_month();
    }
    d_ += n;
    return *this;
  }

  Date & Date::operator-=(int n){
    if(n==0) return *this;
    if(n<0) return operator+=(-n);

    if(n > days_into_year()){
      n-= days_into_year();
      d_ = 31;
      m_ = Dec;
      --y_;

      if(n>1461){
 	int dy4 = n/1461;
 	y_ -= 4*dy4;
 	n = n %1461;
      }while (n>365){
 	n-= 365 + is_leap_year();
 	--y_;}}

    // now n <= 365 and we're sitting  at Dec 31

    while(n>d_){
      n-= d_;
      end_prev_month();
    }
    d_ -= n;
    return *this;
  }


  Date Date::day_zero(1,1,2000);
  void Date::set_zero(const Date &d){ day_zero = d;}
  void Date::set_zero(int m, int d, int yyyy){day_zero = Date(m,d,yyyy);}

  void Date::solve_for_zero(int m, int d, int yyyy, int jdate){
    day_zero = Date(m,d,yyyy);
    day_zero -= jdate;
  }

  Date::print_order Date::po(mdy);
  Date::date_format Date::df(slashes);
  calendar_format Date::month_format(Abbreviations);
  calendar_format Date::day_format(Abbreviations);

  void Date::set_month_format(calendar_format f){
    Date::month_format = f;  }
  void Date::set_day_format(calendar_format f){
    Date::day_format = f;  }
  void Date::set_date_format(Date::date_format f){
    Date::df = f;  }

  std::ostream & Date::display(std::ostream &out)const{
    if(df==script){
      if(po == mdy){
 	display_month(out);
 	out << " " << d_ <<"," << y_;
      }else if(po==dmy){
 	out << d_ <<" ";
 	display_month(out);
 	out << ", " << y_;
      }else if(po==ymd){
 	out << y_ <<", ";
 	display_month(out);
 	out << d_;
      }
      return out;
    }

    char delim(' ');
    if(df == slashes) delim='/';
    if(df==dashes) delim = '-';

    if(po == mdy){
      display_month(out);
      out << delim << day() << delim << year();
    }else if(po==dmy){
      out << day() << delim;
      display_month(out);
      out << delim << year();
    }else if(po == ymd){
      out << year() << delim;
      display_month(out);
      out << delim << day();
    }
    return out;
  }

  void Date::set_print_order(Date::print_order f){ po =f;}

  std::ostream & display(std::ostream &out, day_names d, calendar_format f){
    static const char *Days[] = {"Saturday", "Sunday", "Monday", "Tuesday",
 			   "Wednesday", "Thursday", "Friday"};
    static const char *days[] = {"saturday", "sunday", "monday", "tuesday",
 			   "wednesday", "thursday", "friday"};
    static const char *ds[] = {"sat", "sun", "mon", "tue", "wed", "thu", "fri"};
    static const char *Ds[] = {"Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"};
    if(f==Full) out << Days[d];
    else if(f==full) out << days[d];
    else if(f==Abbreviations) out << Ds[d];
    else if(f==abbreviations) out << ds[d];
    else if(f==numeric){
      uint tmp(d);
      out << tmp;
    }
    return out;
  }


  std::ostream & Date::display_month(std::ostream &out)const{
    static const char* Month_names[]=
      {"", "January", "February", "March", "April", "May", "June", "July",
       "August", "September", "October", "November", "December"};
    static const char * month_names[]=
      {"", "january", "february", "march", "april", "may", "june", "july",
       "august", "september", "october", "november", "december"};
    static const char * Month_abbrevs[]=
      {"", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
       "Oct", "Nov", "Dec"};
    static const char * month_abbrevs[]=
      {"", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
       "oct", "nov", "dec"};
    if(month_format==Full) out << Month_names[m_];
    else if(month_format==full) out << month_names[m_];
    else if(month_format==Abbreviations) out << Month_abbrevs[m_];
    else if(month_format==abbreviations) out << month_abbrevs[m_];
    else out << int(m_);
    return out;
  }
  using std::swap;
  Date guess_date_format(const string &s, char delim){
    std::vector<string> fields = split_delimited(s, delim);
    int m, d, y;
    fields[0]>>m;
    fields[1]>>d;
    fields[2]>>y;

    if(y<=31){
      if(m>12) swap(y,m);
      else if(d>31) swap(y,d);
      else throw bad_date(); // year <=31, but nothing to swap it with
    }// now year is okay;
    assert(y>31);
    if(m>12) swap(d,m);

    assert(m<=12 && m>=1 && d>=1
 	   && d<= days_in_month(month_names(m), is_leap_year(y)));
    return Date(m,d,y);
  }

  string Date::str()const{
    ostringstream os;
    os << *this;
    return os.str();
  }

  std::ostream & operator<<(std::ostream &out, const Date &d){
    d.display(out);
    return out; }

  month_names str2month(const std::string &m){
    if( m=="January" || m=="january"|| m=="Jan" ||  m=="jan"
 	|| m=="01" || m=="1")
      return Jan;
    if( m=="February" ||  m=="february" ||  m=="Feb" ||  m=="feb"
 	|| m=="02" || m=="2")
      return Feb;
    if( m=="March" ||  m=="march" ||  m=="Mar" ||  m=="mar"
 	|| m=="03" || m=="3")
      return Mar;
    if( m=="April" ||  m=="april" ||  m=="Apr" ||  m=="apr"
 	|| m=="04" || m=="4")
      return Apr;
    if( m=="May" ||  m=="may" || m=="05" || m=="5")
      return May;
    if( m=="June" ||  m=="june" ||  m=="Jun" ||  m=="jun"
 	|| m=="06" || m=="6")
      return Jun;
    if( m=="July" ||  m=="july" ||  m=="Jul" ||  m=="jul"
 	|| m=="07" || m=="7")
      return Jul;
    if( m=="August" ||  m=="august" ||  m=="Aug" ||  m=="aug"
 	|| m=="08" || m=="8")
      return Aug;
    if( m=="September" ||  m=="september" ||  m=="Sep" ||  m=="sep"
 	|| m=="09" || m=="9")
      return Sep;
    if(m=="October" || m=="october" ||  m=="Oct" ||  m=="oct" || m=="10")
      return Oct;
    if(m=="November" || m=="november" ||  m=="Nov" ||  m=="nov"||m=="11")
      return Nov;
    if(m=="December" || m=="december" ||  m=="Dec" ||  m=="dec" ||m=="12")
      return Dec;
    throw unknown_month_name(m);
    return unknown_month;
  }

  int operator-(const Date &d1, const Date &d2){
    return d1.julian()-d2.julian();}

  Date mdy2Date(int m, int d, int yyyy){ return Date(m,d,yyyy);}
  Date dmy2Date(int d, int m, int yyyy){ return Date(m,d,yyyy);}
  Date ymd2Date(int yyyy, int m, int d){ return Date(m,d,yyyy);}

  Date mdy2Date(const string &s, char delim){
    std::vector<string> fields = split_delimited(s, delim);
    string m = fields[0];
    int d,y;
    fields[1] >> d;
    fields[2] >> y;
    return Date(str2month(m), d, y);
  }
  Date dmy2Date(const string &s, char delim){
    std::vector<string> fields = split_delimited(s, delim);
    string m = fields[1];
    int d,y;
    fields[0] >> d;
    fields[2] >> y;
    return Date(str2month(m), d, y);
  }
  Date ymd2Date(const string &s, char delim){
    std::vector<string> fields = split_delimited(s, delim);
    string m = fields[1];
    int d,y;
    fields[2] >> d;
    fields[0] >> y;
    return Date(str2month(m), d, y);
  }


  //============================================================
  timepoint::timepoint() : dt_(), t_(0){}
  timepoint::timepoint(const Date &d, const string &hms, char delim)
    : dt_(d), t_(hms2t(hms,delim)) {}

  timepoint::timepoint(double julian_time){
    double date_part;
    t_ = modf(julian_time, &date_part);
    long dp( static_cast<long>(date_part));
    dt_ = Date(dp);
  }

  timepoint::timepoint(const string &mdy, const string &hms):
    dt_(mdy), t_(hms2t(hms)){
  }
  timepoint::timepoint(const timepoint & rhs) : dt_(rhs.dt_), t_(rhs.t_){}

  timepoint & timepoint::operator=(const timepoint &rhs){
    if(&rhs==this) return *this;
    dt_ = rhs.dt_;
    t_ = rhs.t_;
    return *this;
  }

  bool  timepoint::operator==(const timepoint &rhs)const{
    if(&rhs==this) return true;
    if(t_ != rhs.t_) return false;
    if(dt_ !=rhs.dt_) return false;
    return true;  }
  bool timepoint::operator!=(const timepoint &rhs)const{
    return !operator==(rhs);}
  bool timepoint::operator<(const timepoint &rhs)const{
    if(dt_ > rhs.dt_) return false;
    if(dt_ == rhs.dt_) return t_ < rhs.t_;
    return true;  }
  bool timepoint::operator<=(const timepoint &rhs)const{
    return (*this)==(rhs) || (*this)<rhs ;  }
  bool timepoint::operator>(const timepoint &rhs)const{
    return ! ((*this)<= rhs);}
  bool timepoint::operator>=(const timepoint &rhs)const{
    return ! ((*this) < rhs);}


  const double timepoint::seconds_in_day(86400.0);
  const double timepoint::minutes_in_day(1440.0);
  const double timepoint::hours_in_day(24.0); // duh
  const double timepoint::hour_size(0.04166667); // 1/24
  const double timepoint::minute_size(0.000694444444444444);
  const double timepoint::second_size(1.15740740740741e-05);

  double timepoint::time_left_in_day()const{ return 1.0-t_;  }

  unsigned short timepoint::hour()const{
    double x;
    modf(t_*hours_in_day, &x);
    return static_cast<unsigned short>(x);  }

  unsigned short timepoint::minute()const{
    double x;
    modf(t_*minutes_in_day, &x);
    x-= hour()*60;
    return static_cast<unsigned short>(x);  }

  unsigned short timepoint::second()const{
    double x;
    modf(t_*seconds_in_day, &x);
    x-= hour()*3600;
    x-= minute()*60;
    return static_cast<unsigned short>(x);  }

  unsigned short timepoint::hours_left_in_day()const{
    return 23-hour();  }
  unsigned short timepoint::minutes_left_in_hour()const{
    return 59-minute();}
  unsigned short timepoint::seconds_left_in_minute()const{
    return 59-second();}

  timepoint & timepoint::operator+=(double ndays){
    if(ndays==0) return *this;
    if(ndays <0) return operator-=(-ndays);
    if(ndays>1){
      double days;
      ndays = modf(ndays, &days);
      dt_ += static_cast<int>(days); }
    assert(ndays >=0.0 && ndays <1.0);
    if(ndays > time_left_in_day()){
      ++dt_;
      t_ = ndays - time_left_in_day();
    }else t_+=ndays;
    return *this;
  }

  timepoint & timepoint::operator-=(double ndays){
    if(ndays==0) return *this;
    if(ndays<0) return operator+=(-ndays);
    if(ndays >1){
      double days;
      ndays = modf(ndays, &days);
      dt_ -= static_cast<int>(days);
    }
    assert(ndays >= 0.0 && ndays <1.0);
    if(ndays > t_){
      --dt_;
      t_ = 1.0-ndays;
    }else t_-= ndays;
    return *this;
  }

  Date timepoint::date()const{return dt_;}

  char timepoint::date_time_sep_(' ');
  char timepoint::hms_sep_(':');
  void timepoint::date_time_sep(char c){
    timepoint::date_time_sep_=c;}
  void timepoint::hms_sep(char c){
    timepoint::hms_sep_=c;}


  double timepoint::hms2t(const string &hms, char delim)const{
    std::vector<string> fields = split_delimited(hms,delim);
    istringstream hr_str(fields[0]);
    double hr;
    hr_str >> hr;
    double ans  = hr/hours_in_day;

    istringstream min_str(fields[1]);
    double mins;
    min_str >> mins;
    ans+= mins/minutes_in_day;

    if(fields.size()>=3){
      istringstream sec_str(fields[2]);
      double sec;
      sec_str >> sec;
      ans+= sec/seconds_in_day;
    }
    return ans;
  }

  ostream & operator<<(ostream &out, const timepoint &tm){
    out << tm.dt_<<timepoint::date_time_sep_
 	<< tm.hour()<< timepoint::hms_sep_
 	<< tm.minute() << timepoint::hms_sep_
 	<< tm.second();
    return out;
  }

}
