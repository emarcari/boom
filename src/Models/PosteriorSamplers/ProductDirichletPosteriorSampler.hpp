/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_PRODUCT_DIRICHLET_POSTERIOR_SAMPLER_HPP
#define BOOM_PRODUCT_DIRICHLET_POSTERIOR_SAMPLER_HPP

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/ProductDirichletModel.hpp>

namespace BOOM{
class ProductDirichletPosteriorSampler
    : public PosteriorSampler{
 public:

  // template constructor is needed for polymorphic vectors of models
  template <class VECmodel, class SCALmodel>
  ProductDirichletPosteriorSampler(ProductDirichletModel *m,
                                   std::vector<Ptr<VECmodel> > phi,
                                   std::vector<Ptr<SCALmodel> > alpha,
                                   double min_nu=0)
      : m_(m),
        phi_row_prior_(phi.begin(), phi.end()),
        alpha_row_prior_(alpha.begin(), alpha.end()),
        min_nu_(min_nu)
  {}

  virtual void draw();
  virtual double logpri()const;
 private:
  ProductDirichletModel *m_;
  // phi_row_prior_ pretty much has to be Dirichlet
  std::vector<Ptr<VectorModel> > phi_row_prior_;
  std::vector<Ptr<DoubleModel> > alpha_row_prior_;
  double min_nu_;
};
}
#endif// BOOM_PRODUCT_DIRICHLET_POSTERIOR_SAMPLER_HPP
