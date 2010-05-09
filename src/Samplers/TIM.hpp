/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#ifndef BOOM_TIM_HPP
#define BOOM_TIM_HPP
#include <Samplers/Sampler.hpp>
#include <Samplers/MetropolisHastings.hpp>
#include <numopt.hpp>

namespace BOOM{

  // TIM stands for Tailored Independence Metropolis.  Use it when the
  // target function is approximately the log of a multivariate normal
  // distribution.
  class TIM : public MetropolisHastings{
  public:

    TIM(boost::function<double(const Vec &,Vec &, Mat &,int)> logf,
        int dim,
        double nu = 3);

    TIM(const BOOM::Target & logf,
        const BOOM::dTarget & dlogf,
        const BOOM::d2Target & d2logf,
        int dim,
        double nu = 3);
    virtual Vec draw(const Vec &old);

    // in the typical use case the mode is located each iteration.  If
    // you want to avoid locating the mode use fix_mode(true).  To
    // turn mode location back off again use fix_mode(false).
    void fix_mode(bool yn = true);
    bool locate_mode(const Vec & old);
    const Vec & mode()const;
    const Spd & ivar()const;
  private:
    void report_failure(const Vec &old);
    Ptr<MvtIndepProposal> create_proposal(int dim, double nu = 3);

    Ptr<MvtIndepProposal> prop_;
    BOOM::Target f_;
    BOOM::dTarget df_;
    BOOM::d2Target d2f_;
    Vec cand_;
    Vec g_;
    Spd H_;
    bool mode_is_fixed_;
    bool mode_has_been_found_;
  };
}
#endif
