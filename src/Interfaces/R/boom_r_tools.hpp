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

#ifndef BOOM_R_TOOLS_HPP_
#define BOOM_R_TOOLS_HPP_

#include <string>

#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>

//======================================================================
// Client code must:
// #include <Rinternals.h>
// which contains some evil macros.  Always make it the last #include,
// and never include it in a header file.
//
// Note that the functions listed here throw exceptions.  Code that
// uses them should be wrapped in a try-block where the catch
// statement catches the exception and calls Rf_error() with an
// appropriate error message.  The functions handle_exception(), and
// handle_unknown_exception (in handle_exception.hpp), are suitable
// defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
//======================================================================

struct SEXPREC;  // Defined in <Rinternals.h>

namespace BOOM{
  typedef ::SEXPREC* SEXP;

  // Returns list[[name]] if a list element with that name exists.
  // Returns R_NilValue otherwise.
  SEXP getListElement(SEXP list, const std::string &name);

  // Extract the names from a list.  If the list has no names
  // attribute a vector of empty strings is returned.
  std::vector<std::string> getListNames(SEXP list);

  // Set the names attribute of 'list' to match 'list_names'.  BOOM's
  // error reporting mechanism is invoked if the length of 'list' does
  // not match the length of 'list_names'.
  // Returns 'list' with the new and improved set of 'list_names'.
  SEXP setListNames(SEXP list, const std::vector<std::string> &list_names);

  // Creates a new list with the contents of the original 'list' with
  // new_element added.  The names of the original list are copied,
  // and 'name' is appended.  The original 'list' is not modified, so
  // it is possible to write:
  // my_list = appendListElement(my_list, new_thing, "blah");
  // Two things to note:
  // (1) The output is in new memory, so it is not PROTECTED by default
  // (2) Each time you call this function all the list entries are
  //     copied, and (more importantly) new space is allocated, so
  //     you're better off creating the list you want from the
  //     beginning if you can.
  SEXP appendListElement(SEXP list, SEXP new_element, const std::string &name);

  // Returns the class attribute of the specified R object.  If no
  // class attribute exists an empty vector is returned.
  std::vector<std::string> GetS3Class(SEXP object);

  // Returns a pair, with .first set to the number of rows, and
  // .second set to the number of columns.  If the argument is not a
  // matrix then R's error() function will be called.
  std::pair<int, int> GetMatrixDimensions(SEXP matrix);

  // If 'my_list' contains a character vector named 'name' then the
  // first element of that character vector is returned.  If not then
  // R's 'error()' function is called.
  std::string GetStringFromList(SEXP my_list, const std::string &name);

  // If 'my_vector' is a numeric vector, it is converted to a BOOM::Vec.
  // Otherwise R's error() function is called.
  Vec ToBoomVector(SEXP my_vector);

  // If 'r_matrix' is an R matrix, it is converted to a BOOM::Mat.
  // Otherwise R's error() function is called.
  Mat ToBoomMatrix(SEXP r_matrix);

  // If 'my_matrix' is an R matrix, it is converted to a BOOM::Spd.  If
  // the conversion fails then R's error() function is called.
  Spd ToBoomSpd(SEXP my_matrix);

  // If 'my_vector' is an R logical vector, then it is converted to a
  // std::vector<bool>.  Otherwise R's error() function is called.
  std::vector<bool> ToVectorBool(SEXP my_vector);

  // Convert a BOOM vector or matrix to its R equivalent.  Less type
  // checking is needed for these functions than in the other
  // direction because we know the type of the input.
  SEXP ToRVector(const Vec &boom_vector);
  SEXP ToRMatrix(const Mat &boom_matrix);
}

#endif  // BOOM_R_TOOLS_HPP_
