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
#include <Models/Glm/PosteriorSamplers/BinomialLogitSamplerTim.hpp>
#include <boost/bind.hpp>

namespace BOOM{

  typedef BinomialLogitSamplerTim BLST;
BLST::BinomialLogitSamplerTim(BinomialLogitModel *m,
                              Ptr<MvnBase> pri,
                              bool mode_is_stable,
                              double nu)
      : m_(m),
        pri_(pri),
        sam_(boost::bind(&BLST::logp, this, _1),
             boost::bind(&BLST::dlogp, this, _1, _2),
             boost::bind(&BLST::d2logp, this, _1, _2, _3),
             nu)
  {
    if(mode_is_stable) sam_.fix_mode();
  }

  void BLST::draw(){
    Vec beta = sam_.draw(m_->Beta());
    m_->set_Beta(beta);
  }

  double BLST::logpri()const{
    return pri_->logp(m_->Beta());
  }

  double BLST::Logp(const Vec &beta, Vec &g, Mat &h, int nd)const{
    double ans = pri_->Logp(beta, g, h, nd);
    Vec *gp = nd >0 ? &g : 0;
    Mat *hp = nd >1 ? &h : 0;
    ans += m_->log_likelihood(beta, gp, hp, false);
    return ans;
  }

  double BLST::logp(const Vec &beta)const{
    Vec g;
    Mat h;
    return Logp(beta,g,h,0);
  }

  double BLST::dlogp(const Vec &beta, Vec &g)const{
    Mat h;
    return Logp(beta, g, h, 1);
  }

  double BLST::d2logp(const Vec &beta, Vec &g, Mat &h)const{
    return Logp(beta, g, h, 2);
  }

}
