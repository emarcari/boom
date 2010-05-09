/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_SPD_STORAGE_HPP
#define BOOM_SPD_STORAGE_HPP

#include <Models/DataTypes.hpp>
#include <boost/function.hpp>

namespace BOOM{
  namespace SPD{
    class Storage{
    public:
      Storage(bool current=false);
      Storage(const Storage &rhs);
      virtual ~Storage();
      virtual Storage * clone()const=0;

      virtual uint dim()const=0;
      virtual uint size(bool minimal=true)const;

      bool current()const;
      void signal();
      void set_current();
      boost::function<void(void)> create_observer();

      void add_observer(boost::function<void(void)>);
    private:
      bool current_;
      void observe_changes();

      std::vector<boost::function<void(void)> >signals_;
    };
    //------------------------------------------------------------
    class SpdStorage;
    class CholStorage : public Storage{
    public:
      CholStorage();
      CholStorage(const Spd &S);
      CholStorage(const CholStorage &rhs);
      CholStorage * clone()const;
      virtual uint dim()const;
      void set(const Mat &L, bool sig=true);
      void refresh(const SpdStorage &);
      const Mat & value()const;
    private:
      Mat L;
    };
    typedef boost::shared_ptr<CholStorage> CholPtr;
    //------------------------------------------------------------
    class SpdStorage : public Storage{
    public:
      SpdStorage();
      SpdStorage(const Spd &S);
      SpdStorage(const SpdStorage &S);
      SpdStorage * clone()const;
      virtual uint dim()const;
      const Spd & value()const;
      void set(const Spd &, bool sig=true);
      void refresh_from_chol(const CholStorage&, bool inv=false);
      void refresh_from_inv(const SpdStorage &, CholStorage &);
    private:
      Spd sig_;
    };
    typedef boost::shared_ptr<SpdStorage> SpdPtr;
  }
  //____________________________________________________________
  class SpdData
    : public DataTraits<Spd>
  {
  public:
    SpdData(uint n, double diag=1.0, bool ivar=false);
    SpdData(const Spd &S, bool ivar=false);
    SpdData(const SpdData &rhs);
    SpdData * clone()const;

    virtual uint size(bool minimal=true)const;
    virtual uint dim()const;
    virtual ostream & display(ostream &out)const;
    //    virtual istream & read(istream &);

    virtual const Spd & value()const;
    virtual void set(const Spd &v, bool sig=true);

    const Spd & var()const;
    const Spd & ivar()const;
    const Mat & var_chol()const;
    const Mat & ivar_chol()const;
    double ldsi()const;

    void set_var(const Spd &, bool signal=true);
    void set_ivar(const Spd &, bool signal=true);
    void set_var_chol(const Mat &L, bool signal=true);
    void set_ivar_chol(const Mat &L, bool signal=true);
    void set_S_Rchol(const Vec &sd, const Mat &L);

    void ensure_ivar_current()const;
    void ensure_var_current()const;
    void ensure_var_chol_current()const;
    void ensure_ivar_chol_current()const;

    //    const Corr & R()const;
    //    const Spd & Rinv()const;
    //    const Vec & sd()const;
    //    void set_R_S(const Corr &R, const Vec &S);

  private:
    typedef boost::shared_ptr<SPD::SpdStorage>  SpdPtr;
    typedef boost::shared_ptr<SPD::CholStorage> CholPtr;
    typedef boost::shared_ptr<SPD::Storage> StoragePtr;

    mutable SpdPtr var_;
    mutable SpdPtr ivar_;
    mutable CholPtr var_chol_;
    mutable CholPtr ivar_chol_;
    //    boost::shared_ptr<SPD::CorrSdStorage> RS_;

    StoragePtr current_rep_;

    void setup_storage();
    void ensure_current(SpdPtr, CholPtr, SpdPtr, CholPtr)const;
    void ensure_chol_current(CholPtr, SpdPtr, CholPtr, SpdPtr)const;
  };
}
#endif// BOOM_SPD_STORAGE_HPP
