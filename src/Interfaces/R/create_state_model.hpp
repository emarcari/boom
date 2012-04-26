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

#ifndef BOOM_R_INTERFACE_CREATE_STATE_MODEL_HPP_
#define BOOM_R_INTERFACE_CREATE_STATE_MODEL_HPP_

#include <Interfaces/R/list_io.hpp>
#include <Models/StateSpace/StateSpaceModelBase.hpp>

//======================================================================
// Note that the functions listed here throw exceptions.  Code that
// uses them should be wrapped in a try-block where the catch
// statement catches the exception and calls Rf_error() with an
// appropriate error message.  The functions handle_exception(), and
// handle_unknown_exception (in handle_exception.hpp), are suitable
// defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
//======================================================================

namespace BOOM{
  namespace RInterface{
    // Factory method for creating a StateModel based on inputs
    // supplied to R.  Returns a smart pointer to the StateModel that
    // gets created.
    // Args:
    //   list_arg: The portion of the state.specification list (that was
    //     supplied to R by the user), corresponding to the state model
    //     that needs to be created
    //   io_manager: A pointer to the object manaaging the R list that
    //     will record (or has already recorded) the MCMC output
    // Returns:
    //   A Ptr to a StateModel that can be added as a component of
    //   state to a state space model.
    Ptr<StateModel> CreateStateModel(SEXP list_arg, RListIoManager *io_manager,
                                     const string &prefix = "");

    // A callback class for recording the final state that the
    // StateSpaceModelBase sampled in an MCMC iteration.
    class FinalStateCallback : public VectorIoCallback {
     public:
      explicit FinalStateCallback(StateSpaceModelBase *model)
          : model_(model) {}
      virtual int dim()const {return model_->state_dimension();}
      virtual Vec get_vector()const { return model_->final_state();}
     private:
      StateSpaceModelBase * model_;
    };

  }  // namespace RInterface
}  // namespace BOOM
#endif  // BOOM_R_INTERFACE_CREATE_STATE_MODEL_HPP_
