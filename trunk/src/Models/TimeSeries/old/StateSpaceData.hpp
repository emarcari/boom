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

#ifndef STATE_SPACE_DATA_HPP
#define STATE_SPACE_DATA_HPP

#include <Models/DataTypes.hpp>
#include "TimeSeries.hpp"
#include <Models/WeightedData.hpp>

namespace BOOM{
  class StateSpaceModel;

  class LatentState : RefCounted
  {
  public:
    LatentState(uint state_size, uint eta_size);
    LatentState(const LatentState &);
    virtual LatentState * clone()const=0;
    virtual ~LatentState(){}
    friend class StateSpaceModel;
    friend class StateSpaceData;

    const Vec & alpha()const;     // actual (imputed) latent state
    const Vec & a()const;
    const Spd & P()const;
    Vec mu()const;        // E(alpha|Y[t-1]) or Y[n]
    Spd Var()const;       // V(alpha|Y[t-1]) or Y[n]

    virtual void set_alpha(const Vec &A);
    virtual void set_a(const Vec &A);
    virtual void set_P(const Spd &P);
    virtual void set_eta(const Vec &eta);
    uint size(bool=true)const{return alpha_.size();}
  private:
    Vec alpha_;   // m x 1 actual state (to be imputed)
    Vec a_;        // m x 1 E(alpha|Y[t-1])
    Vec r;        // m x 1
    Vec eta_;      // r x 1 (r <= m) 'disturbance'
    Spd P_;        // m x m Var(alpha|Y[t-1])
    void resize(uint p);

    friend void intrusive_ptr_add_ref(LatentState *m){
      m->up_count(); }
    friend void intrusive_ptr_release(LatentState *m){
      m->down_count(); if(m->ref_count()==0) delete m; }
  };
  //------------------------------------------------------------
  class VectorLatentState : public LatentState {
  public:
    VectorLatentState(uint state_size, uint eta_size);
    VectorLatentState(const VectorLatentState &rhs);
    VectorLatentState * clone()const;
    virtual void set_eta(const Vec &eta);
    Ptr<VectorData> Err(){return eta_d;}
  private:
    Ptr<VectorData> eta_d;
  };

  class ScalarLatentState : public LatentState {
  public:
    ScalarLatentState(uint state_size);
    ScalarLatentState(const ScalarLatentState &rhs);
    ScalarLatentState * clone()const;
    virtual void set_eta(const Vec &eta);
    Ptr<DoubleData> Err(){return eta_d;}
  private:
    Ptr<DoubleData> eta_d;
  };

  inline Ptr<LatentState> create_state(uint state_size, uint eta_size){
    if(eta_size==1) return new ScalarLatentState (state_size);
    return new VectorLatentState(state_size, eta_size);
  }


  //------------------------------------------------------------
  class StateSpaceData
    : virtual public Data,
      public MarkovLink<StateSpaceData>
  {
  public:
    StateSpaceData(double obs, Ptr<LatentState> state);
    StateSpaceData(const Vec &obs, Ptr<LatentState> state);
    StateSpaceData(double obs, Ptr<StateSpaceData> prev, Ptr<LatentState> s=0);
    StateSpaceData(const Vec & obs, Ptr<StateSpaceData> prev, Ptr<LatentState> s=0);

    StateSpaceData(const StateSpaceData &);
    StateSpaceData * clone()const;

    friend class StateSpaceModel;

    virtual const Vec & obs()const{return obs_->value();}
    virtual const Vec & err()const{return err_->value();}
    virtual const Mat & K()const{return K_;}
    virtual const Spd & Finv()const{return Finv_;}
    virtual void set_obs(const Vec &Obs){obs_->set(Obs);}
    virtual void set_err(const Vec &Err){err_->set(Err);}
    virtual void set_K(const Mat &K){K_ =K;}
    virtual void set_Finv(const Spd &Finv){Finv_ = Finv;}

    const Vec & alpha()const{return state()->alpha();}
    const Vec & a()const{return state()->a();}
    const Spd & P()const{return state()->P();}

    void set_alpha(const Vec &a){state()->set_alpha(a);}
    void set_a(const Vec &a){state()->set_a(a);}
    void set_P(const Spd &P){state()->set_P(P);}

    virtual uint obs_size()const{return obs().size();}
    virtual ostream & display(ostream &out)const{out << obs(); return out;}
    virtual istream & read(istream &in){obs_->read(in); return in;}
    virtual uint size(bool=true)const{return obs_size();}
    const Ptr<LatentState> state()const;
    Ptr<LatentState> state();
    uint state_size()const{return state()->size();}
  private:
    Ptr<LatentState> state_;
    Ptr<VectorData> obs_;      // p x 1 observed value
    Ptr<VectorData> err_;      // p x 1 forecast error:  obs_ -  mu(a_)
    Spd Finv_;     // p x p V(err|Y[t-1])
    Mat K_;        // m x p regression coefficient E(alpha|Y[t]) = a + K*err
  };
  //----------------------------------------------------------------------
  class ScalarStateSpaceData : public StateSpaceData{
  public:
    ScalarStateSpaceData(double obs, Ptr<LatentState> state);
    ScalarStateSpaceData(double obs, Ptr<ScalarStateSpaceData> prev, Ptr<LatentState> state=0);
    ScalarStateSpaceData(const ScalarStateSpaceData &);
    ScalarStateSpaceData * clone()const;

    void set_obs(const Vec &Obs);
    void set_err(const Vec &Err);
    Ptr<DoubleData> ScalarObs(){return obs_;}
    Ptr<DoubleData> ScalarErr(){return err_;}
  private:
    Ptr<DoubleData> obs_;
    Ptr<DoubleData> err_;
  };

  //----------------------------------------------------------------------
  typedef WeightedData<StateSpaceData> WeightedStateSpaceData;

  //----------------------------------------------------------------------

  TimeSeries<StateSpaceData>
  StateSpaceData_series(uint StateSize, uint ObsSize=1, const string &ID="");

  TimeSeries<StateSpaceData>
  StateSpaceData_series(const std::vector<double> &dv, uint StateSize,
			  const string & ID ="");

  TimeSeries<StateSpaceData>
  StateSpaceData_series(const std::vector<Vec> &dv, uint StateSize,
			  const string & ID = "");


  //----------------------------------------------------------------------

  void set_links(std::vector<Ptr<StateSpaceData> > &dv);

  std::vector<Ptr<ScalarStateSpaceData> >
  make_state_space_data(const std::vector<double> &dv, uint StateSize);

  std::vector<Ptr<StateSpaceData> >
  make_state_space_data(const std::vector<Vec> &dv, uint StateSize);

  Vec mean(const TimeSeries<StateSpaceData> &ds);
  Spd var(const TimeSeries<StateSpaceData> &ds);
}
#endif // STATE_SPACE_DATA_HPP
