/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_DAFE_PCR_RWM_HPP
#define BOOM_DAFE_PCR_RWM_HPP
#include <BOOM.hpp>
#include <Models/IRT/Subject.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/VectorModel.hpp>
#include <Samplers/MetropolisHastings.hpp>

namespace BOOM{
  class MvnModel;
  class MH_Proposal;


  namespace IRT{
    class PartialCreditModel;

    class DafePcrRwmItemSampler : public PosteriorSampler{
    public:
      DafePcrRwmItemSampler(Ptr<PartialCreditModel>,
			    Ptr<MvnModel> Prior,
			    double Tdf);
      void draw();
      double logpri()const;
    private:
      Ptr<PartialCreditModel> mod;
      Ptr<MvnModel> prior;
      Ptr<MetropolisHastings> sampler;
      Ptr<MvtRwmProposal> prop;
      //      Ptr<LocationScaleVectorModel> prop_model;

      const double sigsq;  //  = pi^2/6 = 1.64493406684
      Spd xtx, ivar;
      Vec b;

      void get_moments();
      void accumulate_moments(Ptr<Subject>);
    };
    //======================================================================

    class DafePcrRwmSubjectSampler : public PosteriorSampler{
    public:
      DafePcrRwmSubjectSampler(Ptr<Subject>,
			       Ptr<SubjectPrior> Prior,
			       double Tdf);
      void draw();
      double logpri()const;
    private:
      Ptr<Subject> sub;
      Ptr<SubjectPrior> prior;
      Ptr<MetropolisHastings> sampler;
      Ptr<MvtRwmProposal> prop;
      //      Ptr<LocationScaleVectorModel> prop_model;

      const double sigsq;  //  = pi^2/6 = 1.64493406684
      Spd ivar;
      Vec Theta;

      void get_moments();
      void accumulate_moments(std::pair<Ptr<Item>, Response>);
    };

  }
}
#endif // BOOM_DAFE_PCR_RWM_HPP
