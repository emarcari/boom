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

#include <Models/Glm/PosteriorSamplers/BinomialLogitSamplerRwm.hpp>
#include <distributions.hpp>

namespace BOOM{

  namespace{
    inline VectorView get_chunk(Vec &x, int chunk_size, int chunk_number){
      int start = chunk_number * chunk_size;
      int elements_remaining = x.size() - start;
      return VectorView(x, start, std::min(elements_remaining, chunk_size));
    }
    //----------------------------------------------------------------------

    class BinomialLogitLogPosterior{
     public:
      BinomialLogitLogPosterior(BinomialLogitModel *model,
                                Ptr<MvnBase> prior)
          : m_(model),
            prior_(prior)
      {}
      double operator()(const Vec &beta)const{
        return prior_->logp(beta) + m_->log_likelihood(beta, 0, 0);
      }
     private:
      BinomialLogitModel *m_;
      Ptr<MvnBase> prior_;
    };
  }
  BinomialLogitSamplerRwm::BinomialLogitSamplerRwm(BinomialLogitModel *model,
                                                   Ptr<MvnBase> prior,
                                                   double nu)
      :m_(model),
       pri_(prior),
       proposal_(new MvtRwmProposal(Spd(model->xdim(), 1.0), nu)),
       sam_(BinomialLogitLogPosterior(m_, pri_), proposal_)
  {}

  void BinomialLogitSamplerRwm::draw(){
    const std::vector<Ptr<BinomialRegressionData> > &data(m_->dat());
    Spd ivar(pri_->siginv());
    Vec beta(m_->Beta());
    for(int i = 0; i < data.size(); ++i){
      double eta = beta.dot(data[i]->x());
      double prob = plogis(eta);
      ivar.add_outer(data[i]->x(), prob * (1-prob));
    }

    proposal_->set_ivar(ivar);
    beta = sam_.draw(beta);
    m_->set_Beta(beta);
  }

  double BinomialLogitSamplerRwm::logpri()const{
    return pri_->logp(m_->Beta());
  }
  //======================================================================
}
