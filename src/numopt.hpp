/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_NUMOPT_HPP
#define BOOM_NUMOPT_HPP

#include <string>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>
#include <BOOM.hpp>
#include <boost/function.hpp>

namespace BOOM{

  typedef boost::function<double(const Vec &) > Target;
  typedef boost::function<double(const Vec &x, Vec &g) > dTarget;
  typedef boost::function<double(const Vec &x, Vec &g, Mat &H) > d2Target;

  typedef boost::function<double(double) > ScalarTarget;
//   typedef boost::function<double(double, double &) > dScalarTarget;
//   typedef boost::function<double(double, double &, double &)> d2ScalarTarget;


  enum conj_grad_method{ FletcherReeves, PolakRibiere, BealeSorenson};

  double max_nd0(Vec &x, Target tf);
  double max_nd1(Vec &x, Target tf, dTarget dtf, double eps = 1e-5);
  double max_nd2(Vec &x, Vec &g, Mat &h, Target tf, dTarget dtf,
		 d2Target d2tf, double leps = 1e-5);

  double numeric_deriv(const ScalarTarget f, double x);
  double numeric_deriv(const ScalarTarget f, double x,
		       double &dx, double &abs_err);

  Vec numeric_gradient(const Vec &x, Target f, double dx);
  Mat numeric_hessian(const Vec &x, Target f, double dx);
  Mat numeric_hessian(const Vec &x, dTarget df, double dx);

  //--------- Methods: Each includes a full interface and an inline
  //--------- function providing a simpler interface

  double nelder_mead_driver(Vec &x, Vec &y,
			    Target f,
			    double abstol,
			    double intol,
			    double alpha, double beta, double gamma,
			    bool trace, int & fncount, int maxit);

  double nelder_mead(Vec &x, Vec &y,
 		     Target f,
 		     double abstol,
 		     double intol,
 		     double alpha, double beta, double gamma,
 		     bool trace, int & fncount, int maxit);

  inline double nelder_mead(Vec &x, Vec &y,
			    Target f, double intol = 1e-5){
    double abstol = 1e-8;
    double alpha = 1.0;
    double beta = .5;
    double gamma = 2.0;
    bool trace=false;
    int  fncount=0;
    int maxit=500;
    return nelder_mead_driver(x,y,f, abstol,intol,alpha, beta, gamma,
			      trace, fncount, maxit);
  }

  double bfgs(Vec &x, Target target,
	      dTarget  dtarget,
 	      int maxit, double abstol, double reltol,
 	      int &fncount, int & grcount, bool &fail, int trace_freq= -1);

  inline double bfgs(Vec &b, Target target, dTarget dtarget,
 		     double reltol = 1.0e-5){
    int maxit=200,  fncount=0, grcount=0;
    double abstol = 1e-8;
    bool fail=false;
    return bfgs(b,target, dtarget, maxit, abstol, reltol, fncount,
		grcount, fail, -1);
  }

  double conj_grad(Vec &x, Vec &y, Target f,
 		   dTarget df, double abstol, double intol,
 		   conj_grad_method type, bool trace,
 		   int &fcnt, int &gcnt, int maxit);

  inline double conj_grad(Vec &x, Vec &y, Target f,
 			  dTarget df, double intol = 1e-5){
    conj_grad_method type = PolakRibiere ;
    double abstol = 1e-5;
    int fcnt=0, gcnt=0, maxit = 100;
    bool trace=false;
    return conj_grad(x,y,f,df,abstol, intol, type, trace, fcnt, gcnt, maxit);
  }


  double newton_raphson_min(Vec &x,
                            Vec &g,
                            Mat &h,
                            d2Target f,
                            int &function_call_count,
                            double eps,
                            bool & happy_ending);

  inline double newton_raphson_min(Vec &x,
                                   Vec &g,
                                   Mat &h,
                                   d2Target f,
                                   double leps = 1e-5){
    int fc=0;
    bool happy_ending = true;
    return newton_raphson_min(x,g,h,f,fc,leps, happy_ending);
  }

  double simulated_annealing(Vec &x, Target f,
 			     int maxit, int tmax, double ti, bool trace);

  inline double simulated_annealing(Vec &x, Target f){
    int maxit = 1000;
    int tmax = 100;
    double ti = 20; // ???
    bool trace=false;
    return simulated_annealing(x,f,maxit,tmax,ti,trace);
  }

  class too_many_restarts : public std::runtime_error{
   public:
    too_many_restarts()
        : std::runtime_error("too many restarts")
    {}

    too_many_restarts(const string & msg)
        : std::runtime_error(string("too many restarts in ") + msg)
    {}
  };


  class Negate{
  public:
    Negate(Target F) : f(F){}
    double operator()(const Vec &x)const;
  private:
    Target  f;
  };

  class dNegate : public Negate{
  public:
    dNegate(Target F, dTarget dF)
      : Negate(F), df(dF){}
    double operator()(const Vec &x)const{
      return Negate::operator()(x);}
    double operator()(const Vec &x, Vec &g)const;
  private:
    dTarget df;
  };

  class d2Negate : public dNegate{
  public:
    d2Negate(Target f, dTarget df, d2Target  d2F)
      : dNegate(f, df), d2f(d2F){}
    double operator()(const Vec &x)const{
      return Negate::operator()(x);}
    double operator()(const Vec &x, Vec &g)const{
      return dNegate::operator()(x,g);}
    double operator()(const Vec &x, Vec &g, Mat &h)const;
  private:
    d2Target d2f;
  };


  class bad_initial_value : public std::runtime_error{
   public:
    bad_initial_value(const string &alg, const Vec &X)
        : std::runtime_error(write_error_message(alg,X))
    {}
    ~bad_initial_value() throw() {}
    const char * write_error_message(const string &alg, const Vec &X){
      ostringstream out;
      out << "bad initial value: " << X << " in " << alg;
      return out.str().c_str();
    }
  };


  class newton_raphson_failure : public std::exception{
  public:
    string msg;

    newton_raphson_failure(const Vec &x, double ans, const Vec &g, const Mat &h,
 	    int nd, const std::string &Msg){
      ostringstream out;
      out << "Newton-Raphson failure " << Msg << endl
	  << "function value: " << ans << endl
	  << "x=    " << x << endl;

      if(nd>0) out << "g= " << g << std::endl;
      if(nd>1){
 	out << "h= " << endl;
 	out << h << endl;
 	out << "eigenvalues of h (should all be positive):" << endl;
 	out << h.real_evals() << std::endl;
      }
      msg = out.str();
    }

    const char *what()const throw(){return msg.c_str();}
    ~newton_raphson_failure() throw() {}
  };



}
#endif // BOOM_NUMOPT_HPP
