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

#include "IrtModel.hpp"
#include "Item.hpp"
#include "Subject.hpp"
#include "SubjectPrior.hpp"

#include <cstring>
#include <stdexcept>
#include <iomanip>

#include <cpputil/string_utils.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/random_element.hpp>

#include <LinAlg/CorrelationMatrix.hpp>
#include <algorithm>

#include <Models/MvnModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

using std::setw;

namespace BOOM{
  namespace IRT{

    typedef std::vector<IrtModel::ModelTypeName> ModelVec;

    typedef istringstream INS;
    typedef ostringstream OUTS;


    typedef std::vector<string> StringVec;

    inline void set_default_names(StringVec &s){
      for(uint i=0; i<s.size(); ++i){
	OUTS out;
	out << "subscale[" << i << "]";
	s[i] = out.str();
      }
    }
    //------------------------------------------------------------
    IrtModel::IrtModel()
      : subscale_names_(1),
	theta_freq(1),
	item_freq(1),
	R_freq(1),
	niter(0),
	theta_supressed(false),
	subject_subset(0),
	subject_search_helper(new Subject("", 1)),
	item_search_helper(new NullItem)
    {
      set_default_names(subscale_names_);
    }

    //------------------------------------------------------------
    IrtModel::IrtModel(uint nsub)
      : subscale_names_(nsub),
	//	response_prototype(new ordinal_data<uint>),
	theta_freq(1),
	item_freq(1),
	R_freq(1),
	niter(0),
	theta_supressed(false),
	subject_subset(0),
	subject_search_helper(new Subject("", 1)),
	item_search_helper(new NullItem)
    {
      set_default_names(subscale_names_);
    }

    //------------------------------------------------------------
    IrtModel::IrtModel(const StringVec &SubscaleNames)
      : subscale_names_(SubscaleNames),
	theta_freq(1),
	item_freq(1),
	R_freq(1),
	niter(0),
	theta_supressed(false),
	subject_subset(0),
	subject_search_helper(new Subject("", 1)),
	item_search_helper(new NullItem)
    { }

    IrtModel::IrtModel(const IrtModel &rhs)
      : Model(rhs),
	ParamPolicy(rhs),
	DataPolicy(rhs),
	PriorPolicy(rhs)
    {
      throw_exception<std::runtime_error>("need to implement copy constructor for IrtModel");
    }

    //------------------------------------------------------------
    IrtModel * IrtModel::clone()const{ return new IrtModel(*this);}
    //------------------------------------------------------------
    double IrtModel::pdf(Ptr<Subject> s, bool logscale)const{
      const ItemResponseMap &resp(s->item_responses());
      double ans = 0;
      for(IrIterC it = resp.begin(); it!=resp.end(); ++it){
	Ptr<Item> item = it->first;
	Response r = it->second;
      }
      throw_exception<std::runtime_error>("need to implement 'pdf' for IrtModel");
      return logscale ? ans : exp(ans);
    }
    //------------------------------------------------------------

    double IrtModel::pdf(Ptr<Data> dp, bool logscale)const{
      return pdf(DAT(dp), logscale); }
    //------------------------------------------------------------
    void IrtModel::initialize_params(){
      for(ItemIt it = item_begin(); it!=item_end(); ++it){
	(*it)->initialize_params();
      }
    }
    //------------------------------------------------------------
    void IrtModel::set_subscale_names(const StringVec &names){
      subscale_names_ = names;}

    //------------------------------------------------------------
    const StringVec & IrtModel::subscale_names(){
      return subscale_names_;}

    //------------------------------------------------------------
    inline uint find_max_length(const StringVec &v){
      uint n = v.size();
      uint sz = 0;
      for(uint i=0; i<n; ++i) sz = std::max<uint>(sz, v[i].size());
      return sz;
    }

    //------------------------------------------------------------
    ostream & IrtModel::print_subscales(ostream &out , bool nl, bool decorate){
      uint sz = 0;
      string sep = "   ";
      if(decorate){
	sz = find_max_length(subscale_names());
	out << string(2, '-') << sep << string(sz, '-') << endl;
      }

      for(uint i=0; i<nscales(); ++i){
	if(decorate) out << std::setw(2) << i << sep;
	out << subscale_names()[i];
	if(nl) out << endl;
	else out << " ";
      }
      return out;
    }

    //------------------------------------------------------------
    uint IrtModel::nscales()const{return subscale_names_.size();}
    uint IrtModel::nsubjects()const{return subjects_.size(); }
    uint IrtModel::nitems()const{return items.size();}

    //------------------------------------------------------------
    void IrtModel::add_item(Ptr<Item> item){
      items.insert(item);
      ParamPolicy::add_model(item);
    }

    //------------------------------------------------------------
    ItemIt IrtModel::item_begin(){ return items.begin();}
    ItemIt IrtModel::item_end(){ return items.end();}
    ItemItC IrtModel::item_begin()const{ return items.begin();}
    ItemItC IrtModel::item_end()const{ return items.end();}


    //------------------------------------------------------------
    Ptr<Item> IrtModel::find_item(const string &id, bool nag)const{
      item_search_helper->id_ = id;
      ItemItC it = items.lower_bound(item_search_helper);
      if(it==items.end() || (*it)->id()!=id){
	if(nag){
	  ostringstream msg;
	  msg << "item with id "<< id
	      << " not found in IrtModel::find_item";
	  throw_exception<std::runtime_error>(msg.str());
	}
	return Ptr<Item> ();
      }
      return *it;
    }

    //------------------------------------------------------------
    void IrtModel::add_subject(Ptr<Subject> s){
      BOOM::IRT::add_subject(subjects_, s);
      DataPolicy::add_data(s);
      if(!!subject_prior_) subject_prior_->add_data(s);
    }
    //------------------------------------------------------------
    SI IrtModel::subject_begin(){return subjects_.begin();}
    SI IrtModel::subject_end(){return subjects_.end();}
    CSI IrtModel::subject_begin()const{return subjects_.begin();}
    CSI IrtModel::subject_end()const{return subjects_.end();}

    //------------------------------------------------------------
    Ptr<Subject> IrtModel::find_subject(const string &id, bool nag)const{
      subject_search_helper->id_ = id;
      CSI it = std::lower_bound(subject_begin(), subject_end(),
				subject_search_helper, SubjectLess());
      if(it == subject_end() || (*it)->id()!=id){
	if(nag){
	  ostringstream msg;
	  msg << "subject with id "<< id
	      << " not found in IrtModel::find_subject";
	  throw_exception<std::runtime_error>(msg.str());
	}
	return Ptr<Subject>();
      }
      return *it;
    }

    //------------------------------------------------------------
    void IrtModel::set_subject_prior(Ptr<MvnModel> p){
      subject_prior_ = new MvnSubjectPrior(p);
      allocate_subjects();
    }

    void IrtModel::set_subject_prior(Ptr<SubjectPrior> sp){
      subject_prior_ = sp;
      allocate_subjects();
    }

    IrtModel::PriPtr IrtModel::subject_prior(){
      return subject_prior_;}

    void IrtModel::allocate_subjects(){
      if(!subject_prior_) return;
      for(SI s = subject_begin(); s!=subject_end(); ++s){
	subject_prior_->add_data(*s);
      }
    }

    //------------------------------------------------------------

    void IrtModel::theta_output_frequency(uint n){ theta_freq=n;}
    uint IrtModel::theta_output_frequency()const{return theta_freq;}

    void IrtModel::R_output_frequency(uint n){R_freq=n;}
    uint IrtModel::R_output_frequency()const{return R_freq;}

    void IrtModel::item_param_output_frequency(uint n){ item_freq=n;}
    uint IrtModel::item_param_output_frequency()const{return item_freq;}

    uint IrtModel::io_params(IO io_prm){

      //      if(!!progress) progress->update();
      int ans =0;
      if(!theta_supressed){
	ans = io_theta(io_prm);
	if(io_prm==COUNT) return ans; }

      if( niter % item_freq == 0 || io_prm!=WRITE)
	ans =io_item_params(io_prm);

      if(io_prm==COUNT) return ans;

      ans = io_R(io_prm);
      if(niter % R_freq==0 && io_prm==WRITE)
	ans = io_R(FLUSH);

      ++niter;
      return ans;
    }
    //------------------------------------------------------------
    uint IrtModel::io_R(IO io_prm){
      return subject_prior_->io_params(io_prm);
    }

    uint IrtModel::io_theta(IO io_prm){
      uint ans =0;
      uint sz = subject_subset.size();
      if(sz==0){  // subset has not been declared
 	if(niter % theta_freq==0 || io_prm !=WRITE){
 	  for( SI it = subjects_.begin(); it!=subjects_.end(); ++it){
 	    ans = (*it)->io_Theta(io_prm);
 	    if(io_prm==COUNT) return ans;
 	  }}
      }else{  // a subset has been declared
 	for(uint i=0; i<sz; ++i){
 	  ans = subject_subset[i]->io_Theta(io_prm);
 	  if(io_prm==COUNT) return ans;}}
      return ans;
    }

    uint IrtModel::io_item_params(IO io_prm){
      uint ans=0;
      for(ItemIt it = items.begin(); it!=items.end(); ++it){
 	Ptr<Item> I= *it;
 	ans = I->io_params(io_prm);
 	if(io_prm==COUNT) return ans; }
      return ans;
    }

    void IrtModel::supress_theta_output(bool yn){ theta_supressed=yn;}

    void IrtModel::theta_output_set(const std::vector<string> &ids){
      supress_theta_output(false);
      subject_subset.reserve(ids.size());
      for(uint i=0; i<ids.size(); ++i){
 	subject_subset.push_back(find_subject(ids[i]));}}

    //------------------------------------------------------------
    void read_subject_info_file
    (const string &fname, Ptr<IrtModel> m, const char delim){
      ifstream in(fname.c_str());
      while(in){
	string line;
	getline(in, line);
	if(!in || is_all_white(line)) break;
	StringVec fields =
	  (delim ==' ')? split_string(line): split_delimited(line, delim);

	uint nf = fields.size();
	string id=fields[0];
	if(!!m->find_subject(id, false)){
	  OUTS msg;
	  msg << "IrtModel::read_subject_info_file..." << endl
	      << "subject identifiers must be unique" << endl
	      << "offending id: " << id;
	  throw_exception<std::runtime_error>(msg.str().c_str());}

	if(nf==1){
	  NEW(Subject, s)(id, m->nscales());
	  m->add_subject(s);
	}else if(nf>1){
	  Vec x(nf-1);
	  for(uint i=1; i<nf; ++i) INS(fields[i]) >> x[i-1];
	  NEW(Subject,s)(id, m->nscales(), x);
	  m->add_subject(s);
	}else{
	  OUTS out;
	  out << "0 fields in IrtModel::read_subject_info_file";
	  throw_exception<std::runtime_error>(out.str().c_str());
	}
      }
    }

    void read_item_response_file(const string &fname, Ptr<IrtModel> m){
      ifstream in(fname.c_str());
      while(in){
	string line;
	getline(in, line);
	if(!in || is_all_white(line)) break;

	string subject_id;
	string item_id;
	string response_str;
	INS sin(line);
	sin >> subject_id >> item_id >> response_str;
	Ptr<Subject> sub = m->find_subject(subject_id, false);
	if(!sub){
	  sub = new Subject(subject_id, m->nscales());
	  m->add_subject(sub);
	}

	Ptr<Item> item = m->find_item(item_id, false);
	if(!item){
	  OUTS msg;
	  msg << "item " << item_id
	      << " present in IrtModel::read_item_response_file,"<< endl
	      << "but not in IrtModel::read_item_info_file."<< endl;
	  throw_exception<std::runtime_error>(msg.str().c_str());
	}

	Response r = item->make_response(response_str);
	item->add_subject(sub);
	sub->add_item(item,r); // response levels are shared here
      }
    }

    void IrtModel::item_report(ostream &out, uint max_name_width)const{
      uint maxw=0;
      for(ItemItC it = items.begin();
	  it!=items.end(); ++it){
	maxw = std::max<uint>(maxw, (*it)->name().size());
      }
      maxw = std::min(maxw, max_name_width);
      for(ItemItC it = items.begin();
	  it!=items.end(); ++it){
	(*it)->report(out, maxw);
      }
    }
  } // closes namespace IRT
} // closes namespace BOOM
