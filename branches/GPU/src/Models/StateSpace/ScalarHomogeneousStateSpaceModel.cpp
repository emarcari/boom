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

#include <Models/StateSpace/ScalarHomogeneousStateSpaceModel.hpp>
#include <Models/StateSpace/Filters/NoStorageScalarKalmanFilter.hpp>
#include <LinAlg/SubMatrix.hpp>
#include <LinAlg/VectorView.hpp>
#include <LinAlg/ConstVectorView.hpp>


namespace BOOM{
  using LinAlg::SubMatrix;
  typedef ScalarHomogeneousStateSpaceModel SHSSM;

  SHSSM::ScalarHomogeneousStateSpaceModel(double sigsq)
      : obs_(new ZeroMeanGaussianModel(sigsq)),
        filter_(new ScalarHomogeneousKalmanFilter),
        total_state_size_(0),
        total_innovation_size_(0),
        save_state_(false)
  {
    ParamPolicy::add_model(obs_);
  }
//----------------------------------------------------------------------
  SHSSM::ScalarHomogeneousStateSpaceModel(const Vec &y)
      : obs_(new ZeroMeanGaussianModel(1.0)),
        filter_(new ScalarHomogeneousKalmanFilter),
        total_state_size_(0),
        total_innovation_size_(0),
        save_state_(false)
  {
    ParamPolicy::add_model(obs_);
    add_data_series(make_ts(y));
  }
//----------------------------------------------------------------------
  SHSSM::ScalarHomogeneousStateSpaceModel(std::vector<Ptr<StateT> > state)
      : obs_(new ZeroMeanGaussianModel(1.0)),
        filter_(new ScalarHomogeneousKalmanFilter),
        total_state_size_(0),
        total_innovation_size_(0),
        save_state_(false)
  {
    ParamPolicy::add_model(obs_);
    uint ns = state.size();
    for(uint i=0; i<ns; ++i) add_state(state[i]);
  }
//----------------------------------------------------------------------
  SHSSM::ScalarHomogeneousStateSpaceModel(const SHSSM & rhs)
      : Model(rhs),
        DataInfoPolicy(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        obs_(rhs.obs_->clone()),
        total_state_size_(0),
        total_innovation_size_(0),
        save_state_(rhs.save_state_)
  {
    ParamPolicy::add_model(obs_);
    for(uint i=0; i<state_.size(); ++i) add_state(rhs.state_[i]->clone());
  }
//----------------------------------------------------------------------
  SHSSM * SHSSM::clone()const{return new SHSSM(*this);}
//----------------------------------------------------------------------
  Vec SHSSM::forecast(uint n)const{
    if(this->nseries()!=1){
      ostringstream err;
      err << "called forecast(" << n << ") with " << this->nseries()
          << "time series assigned to the model."
          ;
      throw std::runtime_error(err.str());
    }
    return this->forecast(n, dat());
  }
//----------------------------------------------------------------------
  Vec SHSSM::forecast(uint n, const Vec & history)const{
    Ptr<TimeSeries<DoubleData> > h(make_ts(history));
    return this->forecast(n,*h);
  }
//----------------------------------------------------------------------
  Vec SHSSM::forecast(uint n, const TimeSeries<DoubleData> & history)const{
    TimeSeries<DoubleData> complete_data;
    uint m = history.size();
    complete_data.reserve(history.size() + n);
    for(uint i = 0; i<m; ++i) complete_data.add_1(history[i]);
    for(uint i=0; i<n; ++i){
      NEW(DoubleData, y)(0);
      y->set_missing_status(Data::completely_missing);
      complete_data.add_1(y);
    }
    filter_->fwd(complete_data);
    Vec ans(n);
    for(uint i=0; i<n; ++i){
      const Vec & a(filter_->a(m+i));  // need to add.  Check vs. a[-1] a[+1]
      double y = Z_.dot(a);
      ans[i] = y;
    }
    return ans;
  }
//----------------------------------------------------------------------
  SHSSM * SHSSM::add_state(Ptr<StateT> s){
    state_.push_back(s);
    ParamPolicy::add_model(s);
    total_state_size_ += s->state_size();
    total_innovation_size_ += s->innovation_size();
    return this;
  }
//----------------------------------------------------------------------
double SHSSM::pdf(Ptr<Data> dp, bool logscale)const{

  Ptr<TimeSeries<DoubleData> > d(dp.dcast<TimeSeries<DoubleData> >());
  if(!d){
    ostringstream err;
    err << "cannot cast Ptr<Data> to Ptr<TimeSeries<DoubleData> > in "
        << "ScalarHomogeneousStateSpaceModel::pdf" << endl;
    throw std::runtime_error(err.str());
  }
  const TimeSeries<DoubleData> & ts(*d);
  NoStorageScalarKalmanFilter f(Z_, obs_sigsq(), T_, R_, Q_,
                                initial_state_distribution());
  double ans =  f.logp(ts);
  return logscale ? ans : exp(ans);
}
//----------------------------------------------------------------------
  double SHSSM::logp(const Vec & ts)const{

    //    set_filter();  // needs logical constness

    NoStorageScalarKalmanFilter f(Z_, obs_sigsq(), T_, R_, Q_, 
                                  initial_state_distribution());
    // set observers to make sure these matrices are current
    return f.logp(ts);
  }
//----------------------------------------------------------------------
  void SHSSM::set_obs_filename(const string &s){ obs_->set_param_filename(s); }
//----------------------------------------------------------------------
  Mat SHSSM::make_T(){
    if(T_.nrow() != total_state_size_ || T_.ncol()!=total_state_size_)
      T_.resize(total_state_size_, total_state_size_);
    T_ = 0.0;
    uint ns = state_.size();
    uint lo = 0;
    for(uint i=0; i<ns; ++i){
      uint ni = state_[i]->state_size();
      SubMatrix Tsub(T_, lo, lo + ni-1, lo, lo + ni-1);
      Tsub  = state_[i]->T();
      lo += ni;
    }
    return T_;
  }
//----------------------------------------------------------------------
  Mat SHSSM::make_R(){
    // state_ += R_ * innovation
    // so R_ must have state_size_ rows and innovation_size_ columns
    if(R_.nrow()!= total_state_size_ || R_.ncol()!=total_innovation_size_)
      R_.resize(total_state_size_, total_innovation_size_);
    R_ = 0;
    uint ns = state_.size();
    uint clo=0;
    uint rlo=0;
    for(uint i=0; i< ns; ++i){
      uint state_size = state_[i]->state_size();
      uint innovation_size = state_[i]->innovation_size();
      SubMatrix Rsub(R_, rlo, rlo + state_size-1, clo, clo + innovation_size-1);
      Rsub = state_[i]->R();
      rlo+= state_size;
      clo += innovation_size;
    }
    return R_;
  }
//----------------------------------------------------------------------
  Vec SHSSM::make_Z(){
    if(Z_.size() != total_state_size_) Z_.resize(total_state_size_);
    Z_ = 0;
    uint ns = state_.size();
    uint lo=  0;
    for(uint i=0; i<ns; ++i){
      uint state_size = state_[i]->state_size();
      VectorView Zsub(Z_, lo, state_size);
      Zsub = state_[i]->Z();
      lo += state_size;
    }
    return Z_;
  }
//----------------------------------------------------------------------
  Spd SHSSM::make_Q(){
    if(Q_.nrow()!=total_innovation_size_ )
      Q_.resize(total_innovation_size_);
    Q_ = 0;
    uint ns = state_.size();
    uint lo = 0;
    for(uint i=0; i<ns; ++i){
      uint ss = state_[i]->innovation_size();
      SubMatrix Qsub(Q_, lo, lo + ss-1, lo, lo + ss-1);
      Qsub = state_[i]->Q();
      lo += ss;
    }
    return Q_;
  }
//----------------------------------------------------------------------
Mat SHSSM::filter(const Vec & ts)const{
  uint nr = ts.size();
  uint nc = total_state_size_;
  Mat ans(nr,nc);
  filter_->fwd(ts);
  for(uint i=0; i<nr; ++i) ans.row(i) = filter_->a(i);
  return ans;
}

Mat SHSSM::smooth(const Vec & ts)const{
  filter_->fwd(ts);
  filter_->bkwd_smoother();
  uint nr = ts.size();
  uint nc = total_state_size_;
  Mat ans(nr, nc);
  for(uint i=0; i<nr; ++i) ans.row(i) =filter_->a(i);
  return ans;
}

Mat SHSSM::state_mean_smooth(const Vec & ts)const{
  filter_->fwd(ts);
  filter_->state_mean_smoother();
  uint nr = ts.size();
  uint nc = total_state_size_;
  Mat ans(nr, nc);
  for(uint i=0; i<nr; ++i) ans.row(i) =filter_->a(i);
  return ans;
}

//----------------------------------------------------------------------
  double SHSSM::impute_state(){
    set_filter();
    uint ns = nseries();
    clear_state_models();
    obs_->clear_data();
    init_->clear_data();
    double loglike = 0;
    for(uint i=0; i<ns; ++i){
      const TimeSeries<DoubleData> & y(dat(i));
      loglike += filter_->fwd(y);
      const Mat alpha = filter_->bkwd_sampling();
      if(save_state_){
        state_storage_[&y].push_back(alpha);
      }
      Vec prev = alpha.row(0);
      init_->suf()->update_raw(prev);

      double mu = Z().dot(prev);
      double err = y[0]->value() - mu;
      obs_->suf()->update_raw(err);

      for(uint t=1; t< alpha.nrow(); ++t){
        Vec now = alpha.row(t);
        mu = Z().dot(now);
        err = y[t]->value() - mu;
        obs_->suf()->update_raw(err);
        this->observe_state(prev, now);
        prev=now;
      }

    }
    return loglike;
  }
//----------------------------------------------------------------------
  void SHSSM::save_state(bool do_it){save_state_ = do_it; }
//----------------------------------------------------------------------
  const std::vector<Mat> & SHSSM::state()const{ return this->state(dat()); }
//----------------------------------------------------------------------
  const std::vector<Mat>  & SHSSM::state(const TimeSeries<DoubleData> &y)const{
    StateStorage::const_iterator it = state_storage_.find(&y);
    if(it==state_storage_.end()){
      ostringstream err;
      err << "cannot find a state history associated with the specified time series"
          << endl;
      throw std::runtime_error(err.str());
    }
    return it->second;
  }
//----------------------------------------------------------------------
  double SHSSM::obs_sigsq()const{ return obs_->sigsq();}
//----------------------------------------------------------------------
  void SHSSM::set_obs_sigsq(double s2){ obs_->set_sigsq(s2);}
//----------------------------------------------------------------------
  void SHSSM::sample_state_posterior(){
    uint ns = state_.size();
    for(uint i=0; i<ns; ++i) state_[i]->sample_posterior();
  }
//----------------------------------------------------------------------
  const Vec & SHSSM::Z()const{return Z_;}
//----------------------------------------------------------------------
  uint SHSSM::total_state_size()const{ return total_state_size_;}
//----------------------------------------------------------------------
  void SHSSM::observe_state(const Vec & now, const Vec & next){

    // how to deal with first and last values?
    uint nsm = state_.size();
    uint lo = 0;
    for(uint i=0; i<nsm; ++i){
      uint d =  state_[i]->state_size();
      ConstVectorView now_view(now.data()+lo, d, 1);
      ConstVectorView next_view(next.data()+lo, d, 1);
      state_[i]->observe_state(now_view, next_view);
      lo = d;
    }
  }
//----------------------------------------------------------------------
  double SHSSM::obs_df()const{ return obs_->suf()->n(); }
//----------------------------------------------------------------------
  double SHSSM::obs_sum_of_squares()const{ return obs_->suf()->sumsq(); }
//----------------------------------------------------------------------
  Vec SHSSM::simulate_initial_state()const{
    return initial_state_distribution()->sim();}
//----------------------------------------------------------------------
  double SHSSM::log_state_model_prior()const{
    uint ns = state_.size();
    double ans=0;
    for(uint i=0; i<ns; ++i) ans += state_[i]->logpri();
    return ans;
  }
//----------------------------------------------------------------------
Ptr<MvnModel> SHSSM::initial_state_distribution()const{
  if(!!init_) return init_;
  init_ = new MvnModel(total_state_size() );
  return init_;
}

//----------------------------------------------------------------------
  void SHSSM::clear_state_models(){
    uint ns = state_.size();
    for(uint i=0; i<ns; ++i) state_[i]->clear_data();
  }
//----------------------------------------------------------------------
  void SHSSM::set_filter(){
    T_ = make_T();
    R_ = make_R();
    Z_ = make_Z();
    Q_ = make_Q();

    if(!filter_) filter_ = new ScalarHomogeneousKalmanFilter(Z_, obs_sigsq(), T_, R_, Q_,
                                                             initial_state_distribution());
    else{
      filter_->set_matrices(Z_, obs_sigsq(), T_, R_, Q_);
      filter_->set_initial_state_distribution(initial_state_distribution());
    }
  }





}
