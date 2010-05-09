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

#ifndef NESTED_MAP_H
#define NESTED_MAP_H

#include <BOOM.hpp>
#include <map>
#include <vector>
#include <list>

namespace BOOM{
   template <uint LEV, class INDX, class OBJ> class nested;
   template <uint L, uint LEV, class INDX, class OBJ>
   class level_iterator;

   template<uint LEV, class INDX, class OBJ>
   class nest_common{
     // functions and data common to the general nest and its
     // specialization.
   protected:
     OBJ obj_;
     INDX indx_;
     bool obj_assigned;
     const uint nlev_;
   public:

     nest_common();
     nest_common(const nest_common &rhs);
     nest_common & operator=(const nest_common &rhs);
     nest_common & operator=(const OBJ &rhs);

     OBJ &obj();
     operator OBJ();
     INDX & lab();
     bool test_obj()const;
     uint nlev()const;
   };


   //======================================================================
   template <uint LEV, class INDX, class OBJ>
   class nested : public nest_common<LEV, INDX, OBJ>{
     // general implementation.  Specialization takes care of LEV==1
   public:
     typedef typename std::map<INDX, nested<LEV-1, INDX, OBJ> > nest_t;
     typedef typename nest_t::iterator iterator;
     typedef typename nest_t::const_iterator const_iterator;
     typedef typename nest_t::reverse_iterator  reverse_iterator;
   private:
     nest_t nest;
   public:

     nested();
     nested(const nested<LEV, INDX, OBJ> &n);
     ~nested(){}

     //=-=-=-=-= subscripting =-=-=-=-=
     // subscripting by a single INDX returns the next lower level of the
     // tree.  thus one can use:  nested[s1][s2][s3]

     nested<LEV-1, INDX, OBJ> & operator[](const INDX &s);
     const nested<LEV-1, INDX, OBJ> & operator[](const INDX &s) const;

     // subscripting using a vector<INDX> or list<INDX> returns the
     // portion of the tree thatvector<INDX> or list<INDX> refers to,
     // often a terminal leaf.
     OBJ & operator[](std::vector<INDX> ind);
     const OBJ & operator[](std::vector<INDX> ind) const;
     OBJ & operator[](std::list<INDX> &ind);
     const OBJ & operator[](std::list<INDX> &ind) const;

     nested<LEV, INDX, OBJ> &operator=(const OBJ &ob);
     OBJ & first();
     uint size()const;
     uint size(uint n)const;
     inline bool test_branch(INDX s); // does branch s have an assigned object

     //=-=-=-=-= Iterators across root of the  tree =-=-=-=-=
     iterator begin(){return nest.begin();}
     iterator end(){return nest.end();}
     const_iterator begin()const{return nest.begin();}
     const_iterator end()const{return nest.end();}
     template <uint L> level_iterator<L,LEV,INDX,OBJ> lev_begin();
     template <uint L> level_iterator<L,LEV,INDX,OBJ> lev_end();
   };

   //======================================================================
   template <class INDX, class OBJ>
   class nested<1u,INDX,OBJ> : public nest_common<1u, INDX, OBJ> {
   public:
     typedef typename std::map<INDX, OBJ> nest_t;
     typedef typename nest_t::iterator iterator;
     typedef typename nest_t::const_iterator const_iterator;
     typedef typename nest_t::reverse_iterator  reverse_iterator;
   private:
     nest_t nest;
   public:
     nested();
     nested(const nested<1u, INDX, OBJ> &n);
     ~nested(){}

     // subscripting to obtain lower nodes
     OBJ & operator[](const INDX &s);
     const OBJ & operator[](const INDX &s)const;
     OBJ & operator[](std::list<INDX> idx);
     const OBJ & operator[](std::list<INDX> idx)const ;
     OBJ & operator[](std::vector<INDX> idx);
     const OBJ & operator[](std::vector<INDX> idx)const;

     nested<1u, INDX, OBJ> & operator=(const OBJ &ob);
     OBJ & first();
     uint size(uint n)const;
     uint size()const;
     inline bool test_branch(INDX s);  // does branch s contain any data?

     iterator begin(){return nest.begin();}
     iterator end(){return nest.end();}
     const_iterator begin()const {return nest.begin();}
     const_iterator end()const{return nest.end();}
     level_iterator<1u,1u,INDX,OBJ> lev_begin();
     level_iterator<1u,1u,INDX,OBJ> lev_end();
   };
   //======================================================================
   template<uint LEV, class INDX, class OBJ>
   nest_common<LEV, INDX,OBJ>::nest_common():
     obj_assigned(false), nlev_(LEV){}

   template<uint LEV, class INDX, class OBJ>
   nest_common<LEV, INDX,OBJ>::nest_common(const nest_common &rhs):
     obj_(rhs.obj_), indx_(rhs.indx_), obj_assigned(rhs.obj_assigned),
     nlev_(rhs.nlev_){}

   template<uint LEV, class INDX, class OBJ>
   nest_common<LEV, INDX, OBJ> & nest_common<LEV, INDX, OBJ>::operator=
   (const nest_common<LEV, INDX,OBJ> &rhs){
     if(&rhs==this) return *this;
     obj_=rhs.obj_;
     indx_ = rhs.indx_;
     obj_assigned = rhs.obj_assigned;
     return *this; }

   template<uint LEV, class INDX, class OBJ>
   nest_common<LEV, INDX, OBJ> & nest_common<LEV, INDX, OBJ>::operator=
   (const OBJ &ob){
     if(obj_!=ob) obj_ = ob;
     obj_assigned=true;
     return *this; }

   template <uint LEV, class INDX, class OBJ>
   inline
   uint nest_common<LEV, INDX,OBJ>::nlev()const{return nlev_;}

   template <uint LEV, class INDX, class OBJ>
   inline
   INDX & nest_common<LEV, INDX,OBJ>::lab(){return indx_;}

   template <uint LEV, class INDX, class OBJ>
   inline
   nest_common<LEV, INDX,OBJ>::operator OBJ(){return obj_;}

   template <uint LEV, class INDX, class OBJ>
   inline
   OBJ &nest_common<LEV, INDX,OBJ>::obj(){return obj_;}


   template <uint LEV, class INDX, class OBJ>
   inline
   bool nest_common<LEV, INDX,OBJ>::test_obj()const{ return obj_assigned;}

   //======================================================================

   template <uint LEV, class INDX, class OBJ>
   nested<LEV, INDX, OBJ>::nested(): nest_common<LEV, INDX,OBJ>() {}

   template<class INDX, class OBJ>
   nested<1u, INDX, OBJ>::nested() : nest_common<1u, INDX,OBJ>(){}
   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   nested<LEV, INDX, OBJ>::nested(const nested<LEV, INDX, OBJ> &n)
     : nest_common<LEV,INDX,OBJ>(n), nest(n.nest) {}

   template<class INDX, class OBJ>
   nested<1u, INDX, OBJ>::nested(const nested<1u, INDX, OBJ> &n)
     : nest_common<1u, INDX,OBJ>(n), nest(n.nest){}
   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   inline
   bool nested<LEV, INDX, OBJ>::test_branch(INDX s)
   {return nest[s].test_obj();}

   template<class INDX, class OBJ>
   bool nested<1u, INDX, OBJ>::test_branch(INDX s){ return nest.count(s)>0;  }
   // subscripting to obtain lower nodes
   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   inline
   nested<LEV-1, INDX, OBJ> & nested<LEV, INDX,OBJ>::operator[]
   (const INDX &s){ nest[s].lab()=s; return nest[s]; }

   template<class INDX, class OBJ>
   OBJ & nested<1u, INDX, OBJ>::operator[](const INDX &s){ return nest[s];}

   template <uint LEV, class INDX, class OBJ>
   inline
   const nested<LEV-1, INDX, OBJ> &
   nested<LEV, INDX,OBJ>::operator[](const INDX &s) const{
     nest[s].lab()=s; return nest[s]; }

   template<class INDX, class OBJ>
   const OBJ & nested<1u, INDX, OBJ>::operator[](const INDX &s)const{
     return nest[s];}
   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   OBJ &
   nested<LEV, INDX,OBJ>::operator[]
   (std::vector<INDX> ind){
     // organized so that ind = [top]... [bottom]
     std::list<INDX> idx(ind);
     return operator[](idx);}

   template<class INDX, class OBJ>
   OBJ & nested<1u, INDX, OBJ>::operator[](std::vector<INDX> idx){
     return nest[idx[0]];}

   template <uint LEV, class INDX, class OBJ>
   const OBJ &
   nested<LEV, INDX,OBJ>::operator[]
   (std::vector<INDX> ind) const{
     // organized so that ind = [top]... [bottom]
     std::list<INDX> idx(ind);
     return operator[](idx);}

   template<class INDX, class OBJ>
   const OBJ & nested<1u, INDX, OBJ>::operator[]
   (std::vector<INDX> idx)const {
     return nest[idx[0]];}
   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   OBJ &
   nested<LEV, INDX,OBJ>::operator[]
   (std::list<INDX> &ind){
     if(ind.empty()) return nest_common<LEV,INDX,OBJ>::obj_;
     INDX &s = ind.front();
     std::list<INDX> idx(ind);
     idx.pop_front();
     return nest[s].operator[](idx);}

   template <uint LEV, class INDX, class OBJ>
   const OBJ &
   nested<LEV, INDX,OBJ>::operator[]
   (std::list<INDX> &ind) const{
     if(ind.empty()) return nest_common<LEV,INDX,OBJ>::obj_;
     INDX &s = ind.front();
     std::list<INDX> idx(ind);
     idx.pop_front();
     return nest[s].operator[](idx);}

   template<class INDX, class OBJ>
   OBJ & nested<1u, INDX, OBJ>::operator[]
   (std::list<INDX> idx){
     return (idx.empty())?
       nest_common<1u, INDX,OBJ>::obj_:
       nest[idx.front()];}

   template<class INDX, class OBJ>
   const OBJ &
   nested<1u, INDX, OBJ>::operator[]
   (std::list<INDX> idx) const{
     return (idx.empty())?
       nest_common<1u, INDX,OBJ>::obj_:
       nest[idx.front()];}

   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   inline
   nested<LEV, INDX, OBJ> &
   nested<LEV, INDX,OBJ>::operator=
   (const OBJ &ob){
     nest_common<LEV, INDX, OBJ>::operator=(ob);
     return *this;}

   template<class INDX, class OBJ>
   nested<1, INDX, OBJ> & nested<1u, INDX, OBJ>::operator=
   (const OBJ &ob){
     nest_common<1u, INDX, OBJ>::operator=(ob);
     return *this;}
   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   inline
   OBJ & nested<LEV, INDX,OBJ>::first(){
     return nest.begin()->second.first(); }

   template <class INDX, class OBJ>
   inline
   OBJ & nested<1u, INDX,OBJ>::first(){
     return nest.begin()->second; }
   //----------------------------------------------------------------------
   template <uint LEV, class INDX, class OBJ>
   uint nested<LEV, INDX,OBJ>::size(uint n)const {
     if(n > nest_common<LEV, INDX,OBJ>::nlev_) return 0;
     uint ans = 0;
     for(const_iterator it = begin(); it!=end(); ++it){
       ans += it->second.size(n-1); }
     return ans;}

   template<class INDX, class OBJ>
   uint nested<1u, INDX, OBJ>::size(uint n)const{
     if(n!=1) return 0;
     return nest.size(); }

   template <uint LEV, class INDX, class OBJ>
   inline
   uint nested<LEV, INDX,OBJ>::size()const{return size(0);}

   template<class INDX, class OBJ>
   uint nested<1u, INDX, OBJ>::size()const{ return size(1);}

   //======================================================================

   template<uint L, uint LEV, class INDX, class OBJ>
   class level_iterator{
   public:
     typedef nested<L-1, INDX, OBJ> nest_t;
     typedef nested<LEV, INDX, OBJ> host_t;
     typedef nested<L-1, INDX, OBJ> & reference;
     typedef nested<L-1, INDX, OBJ> * pointer;
     typedef typename std::map<INDX,nest_t>::iterator iterator;
   private:
     host_t *host;
     level_iterator<L+1, LEV, INDX, OBJ> parent;
     iterator current;
   public:
     level_iterator();
     level_iterator(host_t *h);
     void set_host(const host_t *h);
     reference operator*();
     pointer operator->();
     level_iterator & operator++();
     level_iterator & operator--();
     level_iterator & first();
     level_iterator & last();
     level_iterator get_first();
     level_iterator get_last();
     bool operator==(const level_iterator &);
     bool operator!=(const level_iterator &);
   };

   // specialization for top level iterator:  no parent
   template<uint LEV, class INDX, class OBJ>
   class level_iterator<LEV, LEV, INDX, OBJ>{
   public:
     typedef nested<LEV-1, INDX, OBJ> nest_t;
     typedef nested<LEV, INDX, OBJ> host_t;
     typedef nested<LEV-1, INDX, OBJ> & reference;
     typedef nested<LEV-1, INDX, OBJ> * pointer;
     typedef typename std::map<INDX,nest_t>::iterator iterator;
   private:
     host_t *host;
     iterator current;
   public:
     level_iterator();
     level_iterator(host_t *h);
     void set_host(const host_t *h);
     reference operator*();
     pointer operator->();
     level_iterator & operator++();
     level_iterator & operator--();
     level_iterator & first();
     level_iterator & last();
     level_iterator get_first();
     level_iterator get_last();

     bool operator==(const level_iterator &);
     bool operator!=(const level_iterator &);
   };

   //---------- specialization for leaf level iterator
   template <uint LEV, class INDX, class OBJ>
   class level_iterator<1u, LEV, INDX, OBJ>{
   public:
     typedef nested<1u, INDX, OBJ> nest_t;
     typedef nested<LEV, INDX, OBJ> host_t;
     typedef OBJ & reference;
     typedef OBJ * pointer;
     typedef typename std::map<INDX, OBJ>::iterator iterator;
   private:
     host_t *host;
     level_iterator<2u, LEV, INDX, OBJ> parent;
     iterator current;
   public:
     level_iterator();
     level_iterator(host_t *h);
     void set_host(const host_t *h);
     reference  operator*();
     pointer operator->();
     level_iterator & operator++();
     level_iterator & operator--();
     level_iterator & first();
     level_iterator & last();
     level_iterator get_first();
     level_iterator get_last();
     bool operator==(const level_iterator &);
     bool operator!=(const level_iterator &);
   };

   //---------- specialization for leaf level iterator
   template <class INDX, class OBJ>
   class level_iterator<1u, 1u, INDX, OBJ>{
   public:
     typedef nested<1u, INDX, OBJ> nest_t;
     typedef nested<1u, INDX, OBJ> host_t;
     typedef OBJ & reference;
     typedef OBJ * pointer;
     typedef typename std::map<INDX, OBJ>::iterator iterator;
   private:
     host_t *host;
     iterator current;
   public:
     level_iterator();
     level_iterator(host_t *h);
     void set_host(const host_t *h);
     reference  operator*();
     pointer operator->();
     level_iterator & operator++();
     level_iterator & operator--();
     level_iterator & first();
     level_iterator & last();
     level_iterator get_first();
     level_iterator get_last();
     bool operator==(const level_iterator &);
     bool operator!=(const level_iterator &);
   };

   //======================================================================
   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L,LEV,INDX,OBJ>::level_iterator():
     host(0), parent(), current(){}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV,LEV,INDX,OBJ>::level_iterator():
     host(0), current(){}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u, LEV, INDX, OBJ>::level_iterator():
     host(0), parent(), current(){}

   template<class INDX, class OBJ>
   level_iterator<1u, 1u, INDX, OBJ>::level_iterator():
     host(0), current(){}
   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L,LEV,INDX,OBJ>::level_iterator
   (nested<LEV,INDX, OBJ> *h):host(h), parent(h){}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV,LEV,INDX,OBJ>::level_iterator
   (nested<LEV,INDX, OBJ> *h):host(h){}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u,LEV,INDX,OBJ>::level_iterator
   (nested<LEV,INDX, OBJ> *h):host(h), parent(h){}

   template<class INDX, class OBJ>
   level_iterator<1u,1u,INDX,OBJ>::level_iterator
   (nested<1u,INDX, OBJ> *h):host(h){}

   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   void
   level_iterator<L,LEV,INDX,OBJ>::set_host
   (const nested<LEV, INDX, OBJ>* h){host=h;}

   template<uint LEV, class INDX, class OBJ>
   void
   level_iterator<LEV,LEV,INDX,OBJ>::set_host
   (const nested<LEV, INDX, OBJ>* h){host=h;}

   template<uint LEV, class INDX, class OBJ>
   void
   level_iterator<1u,LEV,INDX,OBJ>::set_host
   (const nested<LEV, INDX, OBJ>* h){host=h;}

   template<class INDX, class OBJ>
   void
   level_iterator<1u,1u,INDX,OBJ>::set_host
   (const nested<1u, INDX, OBJ>* h){host=h;}

   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   typename level_iterator<L, LEV, INDX, OBJ>::reference
   level_iterator<L, LEV, INDX, OBJ>::operator*(){return current->second;}

   template<uint LEV, class INDX, class OBJ>
   typename level_iterator<LEV, LEV, INDX, OBJ>::reference
   level_iterator<LEV, LEV, INDX, OBJ>::operator*(){return current->second;}

   template<uint LEV, class INDX, class OBJ>
   typename level_iterator<1u, LEV, INDX, OBJ>::reference
   level_iterator<1u, LEV, INDX, OBJ>::operator*(){return current->second;}

   template<class INDX, class OBJ>
   typename level_iterator<1u, 1u, INDX, OBJ>::reference
   level_iterator<1u, 1u, INDX, OBJ>::operator*(){return current->second;}

   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   typename level_iterator<L, LEV, INDX, OBJ>::pointer
   level_iterator<L,LEV,INDX,OBJ>::operator->(){
     return &(current->second);}

   template<uint LEV, class INDX, class OBJ>
   typename level_iterator<LEV, LEV, INDX, OBJ>::pointer
   level_iterator<LEV,LEV,INDX,OBJ>::operator->(){
     return &(current->second);}

   template<uint LEV, class INDX, class OBJ>
   typename level_iterator<1u, LEV, INDX, OBJ>::pointer
   level_iterator<1u,LEV,INDX,OBJ>::operator->(){
     return &(current->second);}

   template<class INDX, class OBJ>
   typename level_iterator<1u, 1u, INDX, OBJ>::pointer
   level_iterator<1u,1u,INDX,OBJ>::operator->(){
     return &(current->second);}

   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   bool
   level_iterator<L, LEV, INDX, OBJ>::operator==(const level_iterator &rhs){
     return current == rhs.current; }


   template<uint LEV, class INDX, class OBJ>
   bool
   level_iterator<LEV, LEV, INDX, OBJ>::operator==(const level_iterator &rhs){
     return current == rhs.current; }

   template<uint LEV, class INDX, class OBJ>
   bool
   level_iterator<1u, LEV, INDX, OBJ>::operator==(const level_iterator &rhs){
     return current == rhs.current; }

   template<class INDX, class OBJ>
   bool
   level_iterator<1u, 1u, INDX, OBJ>::operator==(const level_iterator &rhs){
     return current == rhs.current; }

   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   bool
   level_iterator<L, LEV, INDX, OBJ>::operator!=(const level_iterator &rhs){
     return current != rhs.current; }

   template<uint LEV, class INDX, class OBJ>
   bool
   level_iterator<LEV, LEV, INDX, OBJ>::operator!=(const level_iterator &rhs){
     return current != rhs.current; }

   template<uint LEV, class INDX, class OBJ>
   bool
   level_iterator<1u, LEV, INDX, OBJ>::operator!=(const level_iterator &rhs){
     return current != rhs.current; }

   template<class INDX, class OBJ>
   bool
   level_iterator<1u, 1u, INDX, OBJ>::operator!=(const level_iterator &rhs){
     return current != rhs.current; }
   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L, LEV, INDX, OBJ> &
   level_iterator<L, LEV, INDX, OBJ>::first(){
     parent.first();  // sets parent's iterator to 'begin'
     current = parent->begin();
     return *this; }

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV, LEV, INDX, OBJ> &
   level_iterator<LEV, LEV, INDX, OBJ>::first(){
     current = host->begin(); return *this;}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u, LEV, INDX, OBJ> &
   level_iterator<1u, LEV, INDX, OBJ>::first(){
     parent.first();
     current = parent->begin();
     return *this;}

   template<class INDX, class OBJ>
   level_iterator<1u,1u, INDX, OBJ> &
   level_iterator<1u,1u, INDX, OBJ>::first(){
     current = host->begin(); return *this;}


   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L, LEV, INDX, OBJ>
   level_iterator<L, LEV, INDX, OBJ>::get_first(){
     level_iterator<L, LEV, INDX, OBJ> ans(host);
     return ans.first(); }

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV, LEV, INDX, OBJ>
   level_iterator<LEV, LEV, INDX, OBJ>::get_first(){
     level_iterator<LEV, LEV, INDX, OBJ> ans(host);
     return ans.first(); }

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u, LEV, INDX, OBJ>
   level_iterator<1u, LEV, INDX, OBJ>::get_first(){
     level_iterator<1u, LEV, INDX, OBJ> ans(host);
     return ans.first(); }

   template<class INDX, class OBJ>
   level_iterator<1u, 1u, INDX, OBJ>
   level_iterator<1u, 1u, INDX, OBJ>::get_first(){
     level_iterator<1u, 1u, INDX, OBJ> ans(host);
     return ans.first(); }
   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L, LEV, INDX, OBJ> &
   level_iterator<L, LEV, INDX, OBJ>::last(){
     parent.last();
     --parent;
     current = parent->end();
     ++parent;
     return *this; }

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV, LEV, INDX, OBJ> &
   level_iterator<LEV, LEV, INDX, OBJ>::last(){
     current = host->end(); return *this;}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u, LEV, INDX, OBJ> &
   level_iterator<1u, LEV, INDX, OBJ>::last(){
     parent.last();
     --parent;
     current = parent->end();
     ++parent;
     return *this; }

   template<class INDX, class OBJ>
   level_iterator<1u, 1u, INDX, OBJ> &
   level_iterator<1u, 1u, INDX, OBJ>::last(){
     current = host->end(); return *this;}

   //----------------------------------------------------------------------
   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L, LEV, INDX, OBJ>
   level_iterator<L, LEV, INDX, OBJ>::get_last(){
     level_iterator<L, LEV, INDX, OBJ> ans(host);
     return ans.last();}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV, LEV, INDX, OBJ>
   level_iterator<LEV, LEV, INDX, OBJ>::get_last(){
     level_iterator<LEV, LEV, INDX, OBJ> ans(host);
     return ans.last();}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u, LEV, INDX, OBJ>
   level_iterator<1u, LEV, INDX, OBJ>::get_last(){
     level_iterator<1u, LEV, INDX, OBJ> ans(host);
     return ans.last();}

   template<class INDX, class OBJ>
   level_iterator<1u, 1u, INDX, OBJ>
   level_iterator<1u, 1u, INDX, OBJ>::get_last(){
     level_iterator<1u, 1u, INDX, OBJ> ans(host);
     return ans.last();}


   //----------------------------------------------------------------------
   struct level_iterator_decrement_error{};  // exception class

   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L, LEV, INDX, OBJ> &
   level_iterator<L, LEV, INDX, OBJ>::operator++(){
     ++current;
     if(current != parent->end()) return *this;
     if((++parent) == parent.get_last()) return *this;
     current = parent->begin();
     return *this; }

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV,LEV,INDX,OBJ> &
   level_iterator<LEV,LEV,INDX,OBJ>::operator++()
   {++current; return *this;}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u, LEV, INDX, OBJ> &
   level_iterator<1u, LEV, INDX, OBJ>::operator++(){
     ++current;
     if(current != parent->end()) return *this;
     if((++parent) == parent.get_last()) return *this;
     current = parent->begin();
     return *this; }

   template<class INDX, class OBJ>
   level_iterator<1u,1u,INDX,OBJ> &
   level_iterator<1u,1u,INDX,OBJ>::operator++()
   {++current; return *this;}

   //----------------------------------------------------------------------


   template<uint L, uint LEV, class INDX, class OBJ>
   level_iterator<L, LEV, INDX, OBJ> &
   level_iterator<L, LEV, INDX, OBJ>::operator--(){
     if(current!= parent->begin()){
       --current;
     }else if( parent == parent.get_first()){
       throw level_iterator_decrement_error();
     }else{
       --parent;
       current = parent->end();  // one past the end
       --current;}
     return *this;}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<LEV,LEV,INDX,OBJ> &
   level_iterator<LEV,LEV,INDX,OBJ>::operator--()
   {--current; return *this;}

   template<uint LEV, class INDX, class OBJ>
   level_iterator<1u, LEV, INDX, OBJ> &
   level_iterator<1u, LEV, INDX, OBJ>::operator--(){
     if(current!= parent->begin()){
       --current;
     }else if( parent == parent.get_first()){
       throw level_iterator_decrement_error();
     }else{
       --parent;
       current = parent->end();  // one past the end
       --current;}
     return *this;}

   template<class INDX, class OBJ>
   level_iterator<1u,1u,INDX,OBJ> &
   level_iterator<1u,1u,INDX,OBJ>::operator--()
   {--current; return *this;}

   //======================================================================
   template <uint LEV, class INDX, class OBJ>
   template <uint L>
   level_iterator<L,LEV,INDX,OBJ>
   nested<LEV, INDX, OBJ>::lev_begin(){
     level_iterator<L,LEV,INDX,OBJ> it(this);
     it.first();
     return it; }

   template <uint LEV, class INDX, class OBJ>
   template <uint L>
   level_iterator<L, LEV,INDX,OBJ>
   nested<LEV, INDX, OBJ>::lev_end(){
     level_iterator<L,LEV,INDX,OBJ> it(this);
     it.last();
     return it; }

   template <class INDX, class OBJ>
   level_iterator<1u,1u,INDX,OBJ>
   nested<1u, INDX, OBJ>::lev_begin(){
     level_iterator<1u,1u,INDX,OBJ> it(this);
     it.first();
     return it; }

   template <class INDX, class OBJ>
   level_iterator<1u, 1u,INDX,OBJ>
   nested<1u, INDX, OBJ>::lev_end(){
     level_iterator<1u,1u,INDX,OBJ> it(this);
     it.last();
     return it; }

   //======================================================================

}
#endif // NESTED_MAP_H
