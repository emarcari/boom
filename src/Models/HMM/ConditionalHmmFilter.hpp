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

#ifndef BOOM_CONDITIONAL_HMM_FILTER_HPP
#define BOOM_CONDITIONAL_HMM_FILTER_HPP

#include <LinAlg/Types.hpp>
#include <LinAlg/Matrix.hpp>
#include <cpputil/Ptr.hpp>
#include <Models/TimeSeries/AugmentedTimeSeries.hpp>
#include <Models/DataPair.hpp>

namespace BOOM{
  class Model;
  class Data;
  class ConditionalMarkovModel;
  class CMM_EMC;
  class EmMixtureComponent;
  class MarkovChoiceData;
  class ChoiceData;

    class ConditionalHmmFilter{
    public:
      typedef DataPair<Data,MarkovChoiceData> MCData;
      typedef AugmentedTimeSeries<MCData, ChoiceData> TsType;

      typedef ConditionalMarkovModel CMM;
      typedef MarkovChoiceData MCD;
      typedef DataPair<Data, MCD> XD;

      ConditionalHmmFilter(std::vector<Ptr<Model> >, Ptr<CMM> );
      virtual ~ConditionalHmmFilter(){}
      uint state_space_size()const;

      double initialize(Ptr<ChoiceData>);
      double loglike(const TsType & );
      double fwd(const TsType & );
      virtual void bkwd(const TsType &)=0;
    private:
      std::vector<Ptr<Model> > mix_;
      Ptr<CMM> mark_;
    protected:
      std::vector<Mat> P;
      Vec pi, logp, logpi, one;
      Mat logQ;
    };

    //----------------------------------------------------------------------
    class ConditionalHmmEmFilter
      : public ConditionalHmmFilter{
    public:
      ConditionalHmmEmFilter(std::vector<Ptr<EmMixtureComponent> > ,

                             Ptr<CMM_EMC>);
      virtual void bkwd(const TsType &);
    private:
      std::vector<Ptr<EmMixtureComponent> > mix_;
      Ptr<CMM_EMC> mark_;
    };
    //----------------------------------------------------------------------

    class ConditionalHmmGibbsFilter
      : public ConditionalHmmFilter{
    public:
      ConditionalHmmGibbsFilter(std::vector<Ptr<Model> > , Ptr<CMM> );
      virtual void bkwd(const TsType &);
    private:
      std::vector<Ptr<Model> > mix_;
      Ptr<CMM> mark_;
    };

}
#endif// BOOM_CONDITIONAL_HMM_FILTER_HPP
