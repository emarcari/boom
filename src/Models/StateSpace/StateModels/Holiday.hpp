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

#include <boost/shared_ptr.hpp>
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>
#include <cpputil/Date.hpp>

namespace BOOM{

  class Holiday;

  Holiday * CreateNamedHoliday(const string &holiday_name,
                               int days_before,
                               int days_after);

  class RandomWalkHolidayStateModel :
      public StateModel,
      public ZeroMeanGaussianModel{
   public:
    // time_zero may need to be set
    RandomWalkHolidayStateModel(Holiday *holiday, const Date &time_zero);
    virtual RandomWalkHolidayStateModel * clone()const;
    virtual void observe_state(const ConstVectorView then,
                               const ConstVectorView now,
                               int time_now);

    virtual uint state_dimension()const;
    virtual void simulate_state_error(VectorView eta, int t)const;

    virtual Ptr<SparseMatrixBlock> state_transition_matrix(int t)const;
    virtual Ptr<SparseMatrixBlock> state_variance_matrix(int t)const;
    virtual SparseVector observation_matrix(int t)const;
    virtual Vec initial_state_mean()const;
    virtual Spd initial_state_variance()const;

    void set_initial_state_mean(const Vec &v);
    void set_initial_state_variance(const Spd &Sigma);
    void set_time_zero(const Date &time_zero);

    virtual void set_sigsq(double sigsq);
   private:
    boost::shared_ptr<Holiday> holiday_;
    Date time_zero_;
    Vec initial_state_mean_;
    Spd initial_state_variance_;
    Ptr<IdentityMatrix> identity_transition_matrix_;
    Ptr<ZeroMatrix> zero_state_variance_matrix_;
    std::vector<Ptr<SingleSparseDiagonalElementMatrix> > active_state_variance_matrix_;

  };

  // A HolidayStateModel is like a SeasonalModel for daily data.
  class HolidayStateModel
      : public StateModel,
        public ZeroMeanGaussianModel{
   public:
    HolidayStateModel();
    // Time zero is the date of the first observation in the training
    // data.
    HolidayStateModel(Date time_zero);
    virtual HolidayStateModel * clone()const;

    // Takes ownership of holiday in a boost::shared_ptr.
    void add_holiday(Holiday *holiday);

    // An initialization to be performed once all all holidays have
    // been added.  This will set the model matrices.
    void initialize();

    // Time zero is the date of the first observation in the training
    // data.
    void set_time_zero(const Date &time_zero);

    virtual void observe_state(const ConstVectorView then,
                               const ConstVectorView now,
                               int time_now);

    // This is the number of holiday coefficients (number of days in a
    // year covered by a holiday effect), minus one.
    virtual uint state_dimension()const;

    virtual void simulate_state_error(VectorView eta, int t)const;

    // The transition matrix is the identity if t is not in a holiday
    // window, and a Seasonal matrix if it is.
    virtual Ptr<SparseMatrixBlock> state_transition_matrix(int t)const;

    // The state_variance_matrix is zero if t is not a holiday.  It
    // matches the Seasonal model if t is in a holiday window.
    virtual Ptr<SparseMatrixBlock> state_variance_matrix(int t)const;

    // The result is Zero if t is not part of a holiday window.
    // [1,0,0,...] if it is.
    virtual SparseVector observation_matrix(int t)const;

    void set_initial_state_mean(const Vec &initial_state_mean);
    virtual Vec initial_state_mean()const;
    void set_initial_state_variance(const Spd &initial_state_variance);
    virtual Spd initial_state_variance()const;

    Date date_at_time_t(int t_days_since_time_zero)const;
    bool in_holiday_window(const Date &date)const;
   private:
    // Ensure that the vector of holidays is ordered by the date of
    // earliest effect.
    std::vector<Holiday *> get_active_holidays(const Date &date);

    Date time_zero_;
    std::vector<boost::shared_ptr<Holiday> > holidays_in_one_year_;
    int number_of_days_in_holiday_windows_;
    bool initialized_;

    Vec initial_state_mean_;
    Spd initial_state_variance_;

    // Workspace used to implement date_at_time_t();
    mutable Date date_marker_;
    mutable int t_marker_;

    // Model matrices for use when t is part of an active holiday
    // window.
    Ptr<SeasonalStateSpaceMatrix> active_transition_matrix_;
    Ptr<UpperLeftCornerMatrix> active_state_variance_matrix_;

    // Model matrices for use when t is _not_ part of an active
    // holiday window.
    Ptr<IdentityMatrix> inactive_transition_matrix_;
    Ptr<ZeroMatrix> inactive_state_variance_matrix_;
  };

  // A Holiday is a "bump" in the value of the series that occurs
  // each year.  It differs from a standard seasonal model in that
  // holidays can sometimes move, either because of complicated
  // religious logic (e.g. easter), or because of calendar effects,
  // such as when independence day falls on a weekend people get the
  // closest Monday or Friday off.
  //
  // A Holiday is defined in terms of a window, specified as the Date
  // of the holiday, as well as some number of days before or after.
  // This window might be of different width each year, as holidays
  // sometimes interact with weekends in strange ways.
  class Holiday{
   public:
    // The Holiday constructor needs to know the initial time (t == 0)
    // and the time between observations so that it can know whether
    // it applies for arbitrary t > 0.  The time_scale argument must
    // be a string matching one of the enum values in the private:
    // section.
    virtual ~Holiday(){}

    // The date the holiday occurs on a given year.
    virtual Date date(int year)const=0;

    // Beginning and end of this Holiday's influence window in the
    // given year.  Years are specifed by a 4 digit integer,
    // e.g. 2011.
    virtual Date earliest_influence(int year)const=0;
    virtual Date latest_influence(int year)const=0;
    // Holidays can sometimes exert an influence before or after the
    // date of the actual holiday.  The number of days from the
    // earliest influenced day to the last influenced day is the
    // maximum_window_width.
    virtual int maximum_window_width()const=0;

    virtual bool active(const Date &d)const;
  };

  // An OrdinaryHoliday is a Holiday that kees track of two integers,
  // days_before_ and days_after_, that define the number of days
  // influence is felt before and after the actual holiday date.  This
  // is an implementation detail
  class OrdinaryHoliday : public Holiday{
   public:
    OrdinaryHoliday(int days_before, int days_after);
    virtual Date earliest_influence(int year)const;
    virtual Date latest_influence(int year)const;
    virtual int maximum_window_width()const;
    virtual Date date(int year)const;
    virtual Date compute_date(int year)const = 0;
   private:
    int days_before_;
    int days_after_;
    typedef int Year;
    mutable map<Year, Date> date_lookup_table_;
    mutable map<Year, Date> earliest_influence_by_year_;
    mutable map<Year, Date> latest_influence_by_year_;
  };

  // A FixedDateHoliday is a Holiday that occurs on the same date each
  // year.  FixedDateHolidays can "bridge" if they occur on the
  // weekend.  Need to decide what that means.  For now I'm ignoring it.
  // TODO(stevescott):  handle bridging.
  //
  // A non-fixed date holiday is called a floating holiday.
  class FixedDateHoliday : public OrdinaryHoliday{
   public:
    // month is an integer between 1 and 12.
    FixedDateHoliday(int month, int day_of_month, int days_before = 1,
                     int days_after = 1);
    virtual Date compute_date(int year)const;
   private:
    // MonthNames is an enum in the range 1:12 defined in Date.hpp
    const MonthNames month_name_;
    const int day_of_month_;
  };

  // For floating holidays, the date() function might be expensive to
  // compute over and over again, so we defer computation to a rarely
  // called function compute_date(), and store the results in a lookup
  // table.  This class implements the lookup table logic, and
  // requires its children to implement compute_date().
  class FloatingHoliday : public OrdinaryHoliday{
   public:
    FloatingHoliday(int days_before, int days_after);
  };

  //----------------------------------------------------------------------
  // Specific holidays observed in the US
  class NewYearsDay : public FixedDateHoliday{
   public:
    NewYearsDay(int days_before, int days_after);
  };

  class MartinLutherKingDay : public FloatingHoliday{
   public:
    MartinLutherKingDay(int days_before, int days_after);
    virtual Date compute_date(int year)const;
  };

  class SuperBowlSunday : public FloatingHoliday{
   public:
    SuperBowlSunday(int days_before, int days_after);
    virtual Date compute_date(int year)const;
  };

  class PresidentsDay : public FloatingHoliday{
   public:
    PresidentsDay(int days_before, int days_after);
    virtual Date compute_date(int year)const;
  };

  class ValentinesDay : public FixedDateHoliday{
   public:
    ValentinesDay(int days_before, int days_after);
  };

  class SaintPatricksDay : public FixedDateHoliday{
   public:
    SaintPatricksDay(int days_before, int days_after);
  };

  class USDaylightSavingsTimeBegins : public FloatingHoliday{
   public:
    USDaylightSavingsTimeBegins(int days_before, int days_after);
    virtual Date compute_date(int year)const;
  };

  class USDaylightSavingsTimeEnds : public FloatingHoliday{
   public:
    USDaylightSavingsTimeEnds(int days_before, int days_after);
    virtual Date compute_date(int year)const;
  };

  class EasterSunday : public FloatingHoliday{
   public:
    EasterSunday(int days_before, int days_after);
    virtual Date compute_date(int year)const;
  };

  class IndependenceDay : public FixedDateHoliday{
   public:
    IndependenceDay(int days_before, int days_after);
  };

  class LaborDay : public FloatingHoliday{
   public:
    LaborDay(int days_before, int days_after);
    virtual Date compute_date(int year)const;
  };

  class Halloween : public FixedDateHoliday{
   public:
    Halloween(int days_before, int days_after);
  };

  class Thanksgiving : public FloatingHoliday{
   public:
    Thanksgiving(int days_before, int days_after);
    virtual Date compute_date(int year)const;
   };

  class MemorialDay : public FloatingHoliday{
   public:
    MemorialDay(int days_before, int days_after);
    virtual Date compute_date(int year)const;
   };

  class VeteransDay : public FixedDateHoliday{
   public:
    VeteransDay(int days_before, int days_after);
   };


  // Christmas is special because sometimes there can be very
  // different numbers of shopping days between Thanksgiving and
  // Christmas in different years.
  class Christmas : public FixedDateHoliday{
   public:
    Christmas(int days_before, int days_after);
  };

}
