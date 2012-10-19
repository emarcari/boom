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

#ifndef BOOM_REF_COUNTED_HPP
#define BOOM_REF_COUNTED_HPP

#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>

namespace BOOM{

  class RefCounted{
    unsigned int cnt_;
    boost::mutex ref_count_mutex_;
  public:
    RefCounted(): cnt_(0){}
    RefCounted(const RefCounted &): cnt_(0), ref_count_mutex_() {}

    // If this object is assigned a new value, nothing is done to the
    // reference count, so assignment is a no-op.
    RefCounted & operator=(const RefCounted &rhs) { return *this; }

    virtual ~RefCounted(){}
    void up_count(){
      boost::lock_guard<boost::mutex> lock(ref_count_mutex_);
      ++cnt_;
    }
    void down_count(){
      boost::lock_guard<boost::mutex> lock(ref_count_mutex_);
      --cnt_;
    }
    unsigned int ref_count()const{return cnt_;}
  };

}
#endif // BOOM_REF_COUNTED_HPP
