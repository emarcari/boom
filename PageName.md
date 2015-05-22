# Introduction #

BOOM is a C++ environment for statistical computing, with a heavy emphasis on Bayesian computation.  Broadly speaking, BOOM has two components, which can be thought of as
  * building blocks, and
  * models.
The building blocks are classes and functions for doing low-level things like manipulating matrices and vectors, evaluating probability distributions, and integrating or sampling from functions.  Whenever possible, BOOM uses high quality existing tools to implement the building blocks.  Matrix computations are done using [LAPACK](http://www.netlib.org/lapack/) and [BLAS](http://www.netlib.org/blas/) (e.g. [ATLAS](http://math-atlas.sourceforge.net/)).  Probability distributions are managed using source code from [R](http://www.r-project.org/).
Low level memory management, filesystem interactions, and functional programming are done using tools from the [boost](http://www.boost.org/) libraries.

Models in BOOM are abstractions for managing the relationship between model parameters and data.  A Model is a class that owns one or more of each of the following three things:  parameters, data, and a learning method (such as a posterior sampler).
