/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include <Models/StateSpace/StateModels/Holiday.hpp>
#include <distributions.hpp>

namespace BOOM{
  typedef HolidayStateModel HSM;

  // Some utility classes in an unnamed namespace.
  namespace{
    class HolidayDateCompare{
     public:
      HolidayDateCompare(int year) : year_(year) {}
      bool operator()(boost::shared_ptr<Holiday> h, const Date &d)const{
        return h->earliest_influence(year_) < d;
      }
     private:
      int year_;
    };

    class HolidayOrder {
     public:
      HolidayOrder(int year) : year_(year) {}
      bool operator()(boost::shared_ptr<Holiday> h1,
                      boost::shared_ptr<Holiday> h2)const{
        return h1->earliest_influence(year_) < h2->earliest_influence(year_);
      }
     private:
      int year_;
    };
  } // un-named namespace

  //======================================================================
  typedef RandomWalkHolidayStateModel RWHSM;
  RWHSM::RandomWalkHolidayStateModel(Holiday *holiday, const Date &time_zero)
      : holiday_(holiday),
        time_zero_(time_zero)
  {
    int dim = holiday->maximum_window_width();
    initial_state_mean_.resize(dim);
    initial_state_variance_.resize(dim);
    identity_transition_matrix_ = new IdentityMatrix(dim);
    zero_state_variance_matrix_ = new ZeroMatrix(dim);
    for(int i = 0; i < dim; ++i){
      NEW(SingleSparseDiagonalElementMatrix, variance_matrix)(dim, 1.0, i);
      active_state_variance_matrix_.push_back(variance_matrix);
    }
  }

  RandomWalkHolidayStateModel * RWHSM::clone()const{
    return new RandomWalkHolidayStateModel(*this);}

  void RWHSM::observe_state(const ConstVectorView then,
                            const ConstVectorView now,
                            int time_now){
    Date today = time_zero_ + time_now;
    if(holiday_->active(today)){
      int position = today - holiday_->earliest_influence(today.year());
      double delta = now[position] - then[position];
      suf()->update_raw(delta);
    }
  }

  uint RWHSM::state_dimension()const{
    return holiday_->maximum_window_width();
  }

  void RWHSM::simulate_state_error(VectorView eta, int t)const{
    Date now = time_zero_ + t;
    eta = 0;
    if(holiday_->active(now)){
      int position = now - holiday_->earliest_influence(now.year());
      eta[position] = rnorm(0, sigma());
    }
  }

  Ptr<SparseMatrixBlock>  RWHSM::state_transition_matrix(int t)const{
    return identity_transition_matrix_;
  }

  Ptr<SparseMatrixBlock> RWHSM::state_variance_matrix(int t)const{
    Date now = time_zero_ + t;
    if(holiday_->active(now)){
      int position = now - holiday_->earliest_influence(now.year());
      return active_state_variance_matrix_[position];
    }
    return zero_state_variance_matrix_;
  }

  SparseVector RWHSM::observation_matrix(int t)const{
    Date now = time_zero_ + t;
    SparseVector ans(state_dimension());
    if(holiday_->active(now)){
      int position = now - holiday_->earliest_influence(now.year());
      ans[position] = 1.0;
    }
    return ans;
  }

  void RWHSM::set_sigsq(double sigsq){
    ZeroMeanGaussianModel::set_sigsq(sigsq);
    for(int i = 0; i < active_state_variance_matrix_.size(); ++i){
      active_state_variance_matrix_[i]->set_value(sigsq);
    }
  }

  Vec RWHSM::initial_state_mean()const{
    return initial_state_mean_;
  }

  Spd RWHSM::initial_state_variance()const{
    return initial_state_variance_;
  }

  void RWHSM::set_initial_state_mean(const Vec &v){
    initial_state_mean_ = v;
  }

  void RWHSM::set_initial_state_variance(const Spd &Sigma){
    initial_state_variance_ = Sigma;
  }

  void RWHSM::set_time_zero(const Date &time_zero){
    time_zero_ = time_zero;
  }

  //======================================================================
  HSM::HolidayStateModel()
      : number_of_days_in_holiday_windows_(0),
        t_marker_(0)
  {}

  HSM::HolidayStateModel(Date time_zero)
      : time_zero_(time_zero),
        number_of_days_in_holiday_windows_(0),
        date_marker_(time_zero),
        t_marker_(0)
  {}

  HolidayStateModel * HSM::clone()const{
    return new HolidayStateModel(*this);}

  void HSM::add_holiday(Holiday *holiday){
    boost::shared_ptr<Holiday> pholiday(holiday);
    holidays_in_one_year_.push_back(pholiday);
    number_of_days_in_holiday_windows_ += holiday->maximum_window_width();
    initialize();
  }

  void HSM::set_time_zero(const Date &time_zero){
    time_zero_ = time_zero;
    date_marker_ = time_zero;
    t_marker_ = 0;
  }

  void HSM::observe_state(const ConstVectorView then,
                          const ConstVectorView now,
                          int time_now){
    Date current_date = date_at_time_t(time_now);
    std::vector<Holiday *> active_holidays = get_active_holidays(current_date);
    int n = active_holidays.size();
    for(int i = 0; i < n; ++i){
      ///////////////////////////////////
      // This is the hard part.
      //      active_holidays[i]->observe_state(then, now);
    }
  }

  std::vector<Holiday *> HSM::get_active_holidays(const Date &current_date){
    std::vector<Holiday *> active_holidays;
    for(int i = 0; i < holidays_in_one_year_.size(); ++i){
      Holiday *holiday = holidays_in_one_year_[i].get();
      if(holiday->active(current_date));
      active_holidays.push_back(holiday);
    }
    return active_holidays;
  }

  uint HSM::state_dimension()const{
    return number_of_days_in_holiday_windows_;
  }

  void HSM::simulate_state_error(VectorView eta, int t)const{
    if(eta.size() != state_dimension()){
      ostringstream err;
      err << "Wrong size vector passed to "
          << "HolidayStateModel::simulate_state_error"
          << endl
          << "Expected 'eta' to be of length " << state_dimension()
          << " but eta.size() is " << eta.size() << endl;
      report_error(err.str());
    }
    Date current_date = date_at_time_t(t);
    eta = 0;
    if(in_holiday_window(current_date)){
      eta[0] = rnorm(0, sigma());
    }
  }

  Ptr<SparseMatrixBlock> HSM::state_transition_matrix(int t)const{
    Date current_date = date_at_time_t(t);
    if(in_holiday_window(current_date)){
      return active_transition_matrix_;
    }
    return inactive_transition_matrix_;
  }

  Ptr<SparseMatrixBlock> HSM::state_variance_matrix(int t)const{
    Date current_date = date_at_time_t(t);
    if(in_holiday_window(current_date)){
      return active_state_variance_matrix_;
    }
    return inactive_state_variance_matrix_;
  }

  SparseVector HSM::observation_matrix(int t)const{
    Date current_date = date_at_time_t(t);
    SparseVector ans(state_dimension());
    if(in_holiday_window(current_date)){
      ans[0] = 1;
    }
    return ans;
  }

  void HSM::set_initial_state_mean(const Vec &mean){
    if(mean.size() != state_dimension()){
      ostringstream err;
      err << "Wrong sized argument to HSM::"
          << "set_initial_state_mean." << endl
          << "State dimension is " << state_dimension()
          << ", but mean is of dimension " << mean.size() << "." << endl;
      report_error(err.str());
    }
    initial_state_mean_ = mean;
  }

  Vec HSM::initial_state_mean()const{
    return initial_state_mean_;
  }

  void HSM::set_initial_state_variance(const Spd &variance){
    if(nrow(variance) != state_dimension()){
      ostringstream err;
      err << "Wrong sized argument to HolidayStateModel::"
          << "set_initial_state_variance." << endl
          << "State dimension is " << state_dimension()
          << ", but variance is of dimension " << nrow(variance) << "."
          << endl;
      report_error(err.str());
    }
    initial_state_variance_ = variance;
  }

  Spd HSM::initial_state_variance()const{
    return initial_state_variance_;}

  void HSM::initialize(){
    active_transition_matrix_ = new SeasonalStateSpaceMatrix(
        number_of_days_in_holiday_windows_);
    active_state_variance_matrix_ = new UpperLeftCornerMatrix(
        state_dimension(), 1.0);
    inactive_transition_matrix_ = new IdentityMatrix(state_dimension());
    inactive_state_variance_matrix_ = new ZeroMatrix(state_dimension());
  }

  Date HSM::date_at_time_t(int t)const{
    return time_zero_ + t;
  }

  bool HSM::in_holiday_window(const Date &date)const{
    for (int i = 0; i < holidays_in_one_year_.size(); ++i) {
      if(holidays_in_one_year_[i]->active(date)) return true;
    }
    return false;
  }

  //======================================================================
  // Now we can define a whole bunch of holidays.

  bool Holiday::active(const Date &d)const{
    int year = d.year();
    return d >= earliest_influence(year) && d <= latest_influence(year);
  }

  OrdinaryHoliday::OrdinaryHoliday(int days_before, int days_after)
      : days_before_(days_before),
        days_after_(days_after)
  { assert(days_before >= 0);
    assert(days_after >= 0);
  }

  Date OrdinaryHoliday::earliest_influence(int year)const{
    map<Year, Date>::iterator it = earliest_influence_by_year_.find(year);
    if(it != earliest_influence_by_year_.end()){
      return it->second;
    }
    Date ans = date(year) - days_before_;
    earliest_influence_by_year_[year] = ans;
    return ans;
  }

  Date OrdinaryHoliday::latest_influence(int year)const{
    map<Year, Date>::iterator it = latest_influence_by_year_.find(year);
    if(it != latest_influence_by_year_.end()){
      return it->second;
    }
    Date ans = date(year) - days_before_;
    latest_influence_by_year_[year] = ans;
    return ans;
  }

  int OrdinaryHoliday::maximum_window_width()const{
    return 1 + days_before_ + days_after_;
  }

  Date OrdinaryHoliday::date(int year)const{
    map<Year, Date>::iterator it = date_lookup_table_.find(year);
    if(it != date_lookup_table_.end()){
      return it->second;
    }
    Date ans = compute_date(year);
    date_lookup_table_[year] = ans;
    return ans;
  }

  //----------------------------------------------------------------------
  FixedDateHoliday::FixedDateHoliday(int month,
                                     int day_of_month,
                                     int days_before,
                                     int days_after)
      : OrdinaryHoliday(days_before, days_after),
        month_name_(MonthNames(month)),
        day_of_month_(day_of_month)
  {}

  Date FixedDateHoliday::compute_date(int year)const{
    Date ans(month_name_, day_of_month_, year);
    return ans;
  }

  //----------------------------------------------------------------------
  FloatingHoliday::FloatingHoliday(int days_before, int days_after)
      : OrdinaryHoliday(days_before, days_after)
  {}

  //----------------------------------------------------------------------
  NewYearsDay::NewYearsDay(int days_before, int days_after)
      : FixedDateHoliday(Jan, 1, days_before, days_after)
  {}
  //----------------------------------------------------------------------
  MartinLutherKingDay::MartinLutherKingDay(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  // MLK day is 3rd Monday of January
  Date MartinLutherKingDay::compute_date(int year)const{
    return nth_weekday_in_month(3, Mon, Jan, year);
  }

  //----------------------------------------------------------------------
  PresidentsDay::PresidentsDay(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  // PresidentsDay is 3rd Monday in Feb.
  Date PresidentsDay::compute_date(int year)const{
    return nth_weekday_in_month(3, Mon, Feb, year);
  }

  //----------------------------------------------------------------------
  SuperBowlSunday::SuperBowlSunday(int days_before, int days_after)
       : FloatingHoliday(days_before, days_after)
   {}

  // The super bowl is currently (2011) played on the first sunday in Feb.
  Date SuperBowlSunday::compute_date(int year)const{
    if(year == 2003) return Date(Jan, 26, 2003);
    else if(year == 1989) return Date(Jan, 22, 1989);
    else if(year == 1985) return Date(Jan, 20, 1985);
    else if(year == 1983) return Date(Jan, 30, 1983);
    else if(year == 1980) return Date(Jan, 20, 1980);
    else if(year == 1979) return Date(Jan, 21, 1979);
    else if(year == 1976) return Date(Jan, 18, 1976);
    else if(year == 1972) return Date(Jan, 16, 1972);
    else if(year == 1971) return Date(Jan, 17, 1971);

    if(year >= 2002) {
      // first Sun in Feb
      return nth_weekday_in_month(1, Sun, Feb, year);
    } else if(year >= 1986){
      // Last Sun in Jan
      Date jan31(Jan, 31, year);
      return jan31 - jan31.days_after(Sun);
    } else if(year >= 1979){
      // 4th Sun in Jan
      return nth_weekday_in_month(4, Sun, Jan, year);
    } else if(year >= 1967){
      // 2nd Sunday, not counting new years
      Date jan1(Jan, 1, year);
      if(jan1.day_of_week() == Sun) ++jan1;
      return jan1 + (jan1.days_until(Sun) + 7);
    } else {
      report_error("No SuperBowl before 1967");
    }
    // should never get here
    return Date(Jan, 1, 1000);
  }
  //----------------------------------------------------------------------

  ValentinesDay::ValentinesDay(int days_before, int days_after)
      : FixedDateHoliday(Feb, 14, days_before, days_after)
  {}

  SaintPatricksDay::SaintPatricksDay(int days_before, int days_after)
      : FixedDateHoliday(Mar, 17, days_before, days_after)
  {}

  //----------------------------------------------------------------------
  USDaylightSavingsTimeBegins::USDaylightSavingsTimeBegins(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  Date USDaylightSavingsTimeBegins::compute_date(int year)const{
    if(year <1967){
      report_error("Can't compute USDaylightSavingsTime before 1967.");
    }
    if(year > 2006){
      // Second Sunday in March
      return nth_weekday_in_month(2, Sun, Mar, year);
    } else if (year >= 1987 ) {
      return nth_weekday_in_month(1, Sun, Apr, year);
    }
    return last_weekday_in_month(Sun, Apr, year);
  }
  //----------------------------------------------------------------------
  USDaylightSavingsTimeEnds::USDaylightSavingsTimeEnds(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  Date USDaylightSavingsTimeEnds::compute_date(int year)const{
    if(year <1967){
      report_error("Can't compute USDaylightSavingsTime before 1967.");
    }
    if(year > 2006){
      // Second Sunday in March
      return nth_weekday_in_month(1, Sun, Nov, year);
    }
    return last_weekday_in_month(Sun, Oct, year);
  }

  //----------------------------------------------------------------------
  EasterSunday::EasterSunday(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  Date EasterSunday::compute_date(int year)const{
    // This code was copied off the internet from a random student's
    // homework assignment.  It was able to reproduce easter sunday
    // from 2004 to 2015.  It is claimed to work between 1900 and
    // 2600.
    if(year <= 1900 || year >= 2600){
      report_error("Can only compute easter dates between 1900 and 2600.");
    }
    int a, b, c, d, e, day;
    a = year % 19;
    b = year % 4;
    c = year % 7;
    d = (19 * a + 24) % 30;
    e = (2 * b + 4 * c + 6 * d + 5) % 7;
    day = 22 + d + e;

    MonthNames month_name(Mar);

    if (day > 31) {
      month_name = Apr;
      day = d + e - 9;
      if (year == 1954 || year == 1981 || year == 2049 || year == 2076) {
        day = d + e - 16;
      }
    }

    Date ans(month_name, day, year);
    return ans;
  }

  MemorialDay::MemorialDay(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  // MemorialDay is the last Monday in May
  Date MemorialDay::compute_date(int year)const{
    Date may31(May, 31, year);
    int days_after_monday = may31.days_after(Mon);
    return may31 - days_after_monday;
  }

  IndependenceDay::IndependenceDay(int days_before, int days_after)
      : FixedDateHoliday(Jul, 4, days_before, days_after)
  {}

  //----------------------------------------------------------------------
  LaborDay::LaborDay(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  // Labor day is the first Monday in September.
  Date LaborDay::compute_date(int year)const{
    Date sep1(Sep, 1, year);
    int days_until = sep1.days_until(Mon);
    return sep1 + days_until;
  }

  Halloween::Halloween(int days_before, int days_after)
      : FixedDateHoliday(Oct, 31, days_before, days_after)
  {}

  VeteransDay::VeteransDay(int days_before, int days_after)
      : FixedDateHoliday(Nov, 11, days_before, days_after)
  {}

  //----------------------------------------------------------------------
  Thanksgiving::Thanksgiving(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after)
  {}

  // Thanksgiving is the 4th Thursday in November.
  Date Thanksgiving::compute_date(int year)const{
    Date nov1(Nov, 1, year);
    int days_until_thanksgiving = nov1.days_until(Thu) + 21;
    return nov1 + days_until_thanksgiving;
  }

  Christmas::Christmas(int days_before, int days_after)
      : FixedDateHoliday(Dec, 25, days_before, days_after)
  {}

    // Factory method to create a Holiday given a string containing
    // the holiday name.
  Holiday * CreateNamedHoliday(const string &holiday_name,
                               int days_before,
                               int days_after){
    if(holiday_name=="NewYearsDay"){
      return new NewYearsDay(days_before, days_after);
    } else if(holiday_name=="MartinLutherKingDay"){
      return new MartinLutherKingDay(days_before, days_after);
    } else if(holiday_name=="SuperBowlSunday"){
      return new SuperBowlSunday(days_before, days_after);
    } else if(holiday_name=="PresidentsDay"){
      return new PresidentsDay(days_before, days_after);
    } else if(holiday_name=="ValentinesDay"){
      return new ValentinesDay(days_before, days_after);
    } else if(holiday_name=="SaintPatricksDay"){
      return new SaintPatricksDay(days_before, days_after);
    } else if(holiday_name=="USDaylightSavingsTimeBegins"){
      return new USDaylightSavingsTimeBegins(days_before, days_after);
    } else if(holiday_name=="USDaylightSavingsTimeEnds"){
      return new USDaylightSavingsTimeEnds(days_before, days_after);
    } else if(holiday_name=="EasterSunday"){
      return new EasterSunday(days_before, days_after);
    } else if(holiday_name=="IndependenceDay"){
      return new IndependenceDay(days_before, days_after);
    } else if(holiday_name=="LaborDay"){
      return new LaborDay(days_before, days_after);
    } else if(holiday_name=="Halloween"){
      return new Halloween(days_before, days_after);
    } else if(holiday_name=="Thanksgiving"){
      return new Thanksgiving(days_before, days_after);
    } else if(holiday_name=="MemorialDay"){
      return new MemorialDay(days_before, days_after);
    } else if(holiday_name=="VeteransDay"){
      return new VeteransDay(days_before, days_after);
    } else if(holiday_name=="Christmas"){
      return new Christmas(days_before, days_after);
    }

    ostringstream err;
    err << "Unknown holiday name passed to CreateHoliday:  " << holiday_name;
    report_error(err.str());

    return NULL;
  }

}
