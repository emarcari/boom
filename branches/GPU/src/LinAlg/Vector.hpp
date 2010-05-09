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

#ifndef BOOM_LINALG_VECTOR_HPP
#define BOOM_LINALG_VECTOR_HPP

#include <boost/operators.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>

#include <iosfwd>
#include <vector>
#include <string>
#include <uint.hpp>
#include "VectorConcept.hpp"

namespace BOOM{
  namespace LinAlg{
    class SpdMatrix;
    class Matrix;
    class VectorView;
    class ConstVectorView;

    class Vector
      : public std::vector<double>,
	boost::field_operators<Vector,
			     boost::addable<Vector,double,
                             boost::subtractable<Vector,double,
                             boost::multipliable<Vector,double,
			     boost::dividable<Vector,double> > > > >
    {
    public:
      typedef unsigned int uint;
      typedef std::vector<double> dVector;
      typedef dVector::iterator iterator;
      typedef dVector::const_iterator const_iterator;
      typedef dVector::reverse_iterator reverse_iterator;
      typedef dVector::const_reverse_iterator const_reverse_iterator;

      //--------- constructors, destructor, assigment, operator== ----------
      Vector();
      explicit Vector(uint n, double x=0);
      explicit Vector(const std::string &s);
      Vector(const std::string &s, const std::string &sep);
      Vector(const dVector &d);
      template <class FwdIt> Vector(FwdIt begin, FwdIt end);

      Vector(const Vector &);              // value semantics
      template <class VEC>
      Vector(const VEC &rhs, typename VEC::const_iterator * =0)
	: dVector(rhs.begin(), rhs.end())
      {
	boost::function_requires< VectorConcept<VEC> >();
      }

      virtual ~Vector();

      Vector & operator=(const Vector &);  // value semantics
      Vector & operator=(const double &);  // value semantics
      Vector & operator=(const dVector &);
      Vector & operator=(const VectorView &);
      Vector & operator=(const ConstVectorView &);


      Vector & swap(Vector &);
      bool operator==(const Vector &rhs)const;

      Vector zero()const;  // returns a same sized Vector filled with 0's
      Vector one()const;   // returns a same sized Vector filled with 1's
      Vector & randomize();    // fills the Vector with U(0,1) random numbers
      template <class RNG>
      Vector & randomize(RNG f);

      //-------------- STL vector stuff ---------------------
      double *data();
      const double *data()const;
      uint stride()const{return 1;}
      uint length()const; // same as size()

      //------------ resizing operations
      template <class VEC>
      Vector & concat(const VEC &v);
      Vector & push_back(double x);

      //------------------ checked subscripting -----------------------
      const double & operator()(uint n)const;
      double & operator()(uint n);

      //---------------- input/output -------------------------
      std::ostream & write(std::ostream &, bool endl=true)const;
      std::istream & read(std::istream &);


      //--------- math ----------------
      Vector & operator+=(const double &x);
      Vector & operator-=(const double &x);
      Vector & operator*=(const double &x);
      Vector & operator/=(const double &x);
      Vector & operator+=(const Vector &y);
      Vector & operator-=(const Vector &y);
      Vector & operator*=(const Vector &y);
      Vector & operator/=(const Vector &y);

      //--------- linear algebra
      Vector & axpy(const Vector &x, double w); // *this += w*x
      Vector & axpy(const VectorView &x, double w); // *this += w*x
      Vector & axpy(const ConstVectorView &x, double w); // *this += w*x
      Vector & add_Xty(const Matrix &X, const Vector &y, double w=1.0);
      // *this += w * X^T *y

      Vector & mult(const Matrix &A, Vector &ans)const;        // v^T A
      Vector mult(const Matrix &A)const;                       // v^T A
      Vector & mult(const SpdMatrix &A, Vector &ans)const;    // v^T A
      Vector mult(const SpdMatrix &A)const;                   // v^T A

      double dot(const Vector &y)const; // dot product ignores lower bounds
      double dot(const VectorView &y)const;
      double dot(const ConstVectorView &y)const;

      double affdot(const Vector &y)const;
      double affdot(const VectorView &y)const;
      double affdot(const ConstVectorView &y)const;
      // affine dot product:  dim(y) == dim(x)-1. ignores lower bounds

      SpdMatrix outer()const;      // x x^t
      void outer(SpdMatrix &ans)const; // ans+=x x^t
      Matrix outer(const Vector &y, double a=1.0)const;  // a*x y^t
      void outer(const Vector &y, Matrix &ans, double a=1.0)const; // ans += axy^t

      Vector & normalize_prob();    // *this/= sum(*this)
      Vector & normalize_logprob(); // *this = exp(*this)/sum(exp(*this))
      Vector & normalize_L2();      // *this /= *this.dot(*this);

      double normsq()const;         // *this.dot(*this);
      double min() const;
      double max() const;
      uint imax()const;  // index of maximal/minmal element
      uint imin()const;
      double sum() const;
      double abs_norm()const;  // sum of absolute values.. faster than sum
      double prod() const;

      Vector & sort();
    private:
      bool inrange(uint n)const{return n< size();}
    };
    //----------------------------------------------------------------------

    template <class FwdIt>          // definition of template constructor
    Vector::Vector(FwdIt Beg, FwdIt End)
      : dVector(Beg, End)
    {}

    template <class VEC>
    Vector & Vector::concat(const VEC &v){
      iterator old_end = end();
      reserve(size() + v.size());
      dVector::insert(end(), v.begin(), v.end());
      return *this;
    }

    template <class RNG>
    Vector & Vector::randomize(RNG f){
      for(uint i=0; i<size(); ++i) (*this)[i] = f();
      return *this;
    }


    //======================= Vector functions ======================
    void permute_Vector(Vector &v, const std::vector<uint> &perm);
    Vector str2vec(const std::vector<std::string> &);
    Vector str2vec(const std::string &line);
    Vector scan_vector(const std::string &fname);

    // operators not covered by boost
    Vector operator/(double a, const Vector &x);
    Vector operator-(double a, const Vector &x);

    // unary transformations
    Vector operator-(const Vector &x); // unary minus
    Vector log(const Vector &x);
    Vector exp(const Vector &x);
    Vector sqrt(const Vector &x);
    Vector pow(const Vector &x, double p);
    Vector pow(const Vector &x, int p);

    inline double sum(const Vector &x){return x.sum();}
    inline double prod(const Vector &x){return x.prod();}
    inline double max(const Vector &x){return x.max();}
    inline double min(const Vector &x){return x.min();}
    inline void swap(Vector &x, Vector &y){x.swap(y);}
    Vector cumsum(const Vector &x);
    // IO
    std::ostream & operator<<(std::ostream & out, const Vector &x);
    std::istream & operator>>(std::istream &, Vector &);
    Vector read_Vector(std::istream &in);

    // size changing operations
    using boost::disable_if;
    using boost::is_arithmetic;
    template <class VEC1, class VEC2>
    Vector concat(const VEC1 &x, const VEC2 &y,
		  typename disable_if<is_arithmetic<VEC1> >::type* =0,
		  typename disable_if<is_arithmetic<VEC2> >::type* =0)
    {
      boost::function_requires< VectorConcept<VEC1> >();
      boost::function_requires< VectorConcept<VEC2> >();
      Vector ans(x);
      return ans.concat(y);
    }

    template <class VEC>
    Vector concat(double x, const VEC &y){
      Vector ans(1,x);
      return ans.concat(y); }

    template <class VEC>
    Vector concat(const VEC &x, double y, VectorConcept<VEC> * =0 ){
      Vector ans(x);
      ans.push_back(y);
      return ans;}

    template <class VEC>
    Vector concat(const std::vector<VEC> &v){
      if(v.size()==0) return Vector();
      Vector ans(v.front());
      for(uint i=1; i<v.size(); ++i) ans.concat(v[i]);
      return ans;
    }

    Vector select(const Vector &v, const std::vector<bool> & inc, uint nvars);
    Vector select(const Vector &v, const std::vector<bool> & inc);
    Vector sort(const Vector &v);
    Vector sort(const VectorView &v);
    Vector sort(const ConstVectorView &v);
  }
}
#endif //BOOM_LINALG_VECTOR_HPP
