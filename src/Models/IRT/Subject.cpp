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

#include "Subject.hpp"
#include "Item.hpp"
#include <Models/Glm/Glm.hpp>
#include <stdexcept>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM{
  namespace IRT{

//     void intrusive_ptr_add_ref(Subject *s){
//       intrusive_ptr_add_ref(dynamic_cast<Model*>(s));}

//     void intrusive_ptr_release(Subject *s){
//     intrusive_ptr_release(dynamic_cast<Model*>(s));}

    Subject::Subject(const string &Id, uint nsub)
      : id_(Id),
	responses_(),
	search_helper(new NullItem),
	Theta_(new VectorParams(nsub, 0.0)),
	x_(),
	prototype()
    {
    }

    Subject::Subject(const string &Id, const Vec & theta)
      :
      id_(Id),
      responses_(),
      search_helper(new NullItem),
      Theta_(new VectorParams(theta)),
      x_(),
      prototype()
    {
    }

    Subject::Subject(const string &Id, uint nsub, const Vec & bg)
      : id_(Id),
	responses_(),
	search_helper(new NullItem),
	Theta_(new VectorParams(nsub, 0.0)),
	x_(bg),
	prototype()
    {
    }

    Subject::Subject(const Subject &rhs)
      : Data(rhs),
	id_(rhs.id_),
	responses_(rhs.responses_),
	search_helper(new NullItem),
	Theta_(rhs.Theta_->clone()),
	x_(rhs.x_),
	prototype(rhs.prototype->clone())
    {}

    Subject * Subject::clone()const{ return new Subject(*this); }
    uint Subject::size(bool)const{
      return responses_.size();}

    uint Subject::Nitems()const{return responses_.size();}
    uint Subject::Nscales()const{return Theta().size();}

    Response Subject::add_item(Ptr<Item> item, Response r){
      //      responses_.insert(std::make_pair(item,r));
      responses_[item] = r;
      return r;
    }

    Response Subject::add_item(Ptr<Item> it, uint resp){
      Response r = new OrdinalData(resp, it->possible_responses_);
      add_item(it,r);
      return r;
    }

    Response Subject::add_item(Ptr<Item> it, const string &resp){
      Response r = new OrdinalData(resp, it->possible_responses_);
      add_item(it,r);
      return r;
    }

    ostream & Subject::display(ostream &out)const{
      out << id();
      if(x_.size()>0) out << x_;
      out << endl;
      return out;

    }

    ostream & Subject::display_responses(ostream &out)const{
      // display Subject_id \t Item_id \t response
      for(IrIterC it = responses_.begin(); it!=responses_.end(); ++it){
	Ptr<Item> item = it->first;
	Response r = it->second;
	out << this->id() << "\t" << item->id() << "\t" ;
	r->display(out) << endl;
      }
      return out;
    }

//     istream & Subject::read(istream & in){
//       Response r = prototype->clone();
//       r->read(in);
//       string item_id;
//       in >> item_id;
//       Ptr<Item> it = find_item(item_id);
//       if(!it){
// 	ostringstream msg;
// 	msg << "item " << item_id << "not found by subject "<< id()
// 	    << " during Subject::read()";
// 	throw_exception<std::runtime_error>(msg.str().c_str());
//       }
//       response(it)->set(r->value());
//       return in;
//     }

    Ptr<Item> Subject::find_item(const string &item_id, bool nag)const{
      search_helper->id_ = item_id;
      IrIterC it = responses_.lower_bound(search_helper);
      if(it==responses_.end() ||  it->first->id()!=item_id){
	if(nag){
	  ostringstream msg;
	  msg << "item with id "<< item_id
	    << " not found in Subject::find_item";
	  throw_exception<std::runtime_error>(msg.str());
	}
	return Ptr<Item>();
      }
      else return it->first;
    }

    double Subject::loglike()const{
      double ans=0;
      for(IrIterC it = responses_.begin(); it!=responses_.end(); ++it){
	Ptr<Item> I = it->first;
	Response resp = it->second;
	ans += I->response_prob(resp, Theta(), true);
      }
      return ans;
    }

    const string & Subject::id()const{return id_;}

    Ptr<VectorParams> Subject::Theta_prm(){
      return Theta_;}
    const Ptr<VectorParams> Subject::Theta_prm()const{
      return Theta_;}

    const Vec & Subject::Theta()const{
      return Theta_prm()->value();}

    void Subject::set_Theta(const Vec &v){
      Theta_prm()->set(v);}

    uint Subject::io_Theta(IO io_prm){ return Theta_prm()->io(io_prm); }

    void Subject::set_Theta_fname(const string &fname){
      Theta_prm()->set_fname(fname); }

    const ItemResponseMap & Subject::item_responses()const{
      return responses_; }

    Response Subject::response(const Ptr<Item> item)const{
      IrIterC it = responses_.find(item);
      if(it==responses_.end()) return Response();
      else return it->second; }

    Spd Subject::xtx()const{
      Spd ans(Nscales(), 0.0);
      Selector inc(Nscales()+1, true);
      inc.drop(0);

      for(IrIterC it = responses_.begin(); it!=responses_.end(); ++it){
	Ptr<Item> item(it->first);
	Vec b = inc.select(item->beta());
	ans.add_outer(b);
      }
      return ans;
    }

     Response Subject::simulate_response(Ptr<Item> it){
       return it->simulate_response(Theta());
     }


  }  // namespace IRT
} // namsepace BOOM
