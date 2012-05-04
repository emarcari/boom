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
#include <cpputil/ProgramOptions.hpp>

namespace BOOM{

typedef ProgramOptions PO;

void PO::add_option_family(const string & name, const string  & desc){
  boost::shared_ptr<OD>  op(new OD(desc));
  option_families_[name] = op;
  option_family_names_.push_back(name);
}


void PO::add_option(const string & option_name,
                    const string & description)
{
  od_.add_options()(option_name.c_str(), description.c_str());
}


void PO::add_option_to_family(const string & fam, const string &op, const string &desc){
  option_families_[fam]->add_options()(op.c_str(), desc.c_str());
}

void PO::process_command_line(int argc, char **argv){
  uint n = option_family_names_.size();
  for(uint i=0; i<n; ++i){
    // doing it this way keeps the options in the same order the user
    // entered them

    string name = option_family_names_[i];
        std::cerr << name << std::endl;
    OD op(*option_families_[name]);
    od_.add(op);
  }
  Svec args(argv+1, argv+argc);
  process_command_line(args);
}

void PO::process_command_line(const Svec & args){
  po::store(po::command_line_parser(args).options(od_).run(), vm);
}

void PO::process_cfg_file(const string &cfg_file_name){
  const string & fname(cfg_file_name);
  ifstream in(fname.c_str());
  if(!in){
    cerr << "No configuration file " << fname << " found."
         << endl;
    return;
  }
  string line;
  Svec args;
  StringSplitter split;
  while(in){
    getline(in,line);
    if(!in) break;
    line = strip(line,"#\r\n");
    if(is_all_white(line)) continue;
    Svec fields = split(line);
    trim_white_space(fields);
    args.reserve(args.size() + fields.size());
    std::copy(fields.begin(), fields.end(), back_inserter(args));
  }

  try{
    po::store(po::command_line_parser(args).options(od_).run(),vm);
  }catch( std::exception &e){
    cerr << "***************************************" << endl
         << "error parsing the configuration file." << endl
         << "***************************************" << endl;
    throw;
  }
}

ostream &  PO::print(ostream &out)const{
  out << od_ << endl;
  return out;
}

ostream & operator<<(ostream &out, const PO & op){return op.print(out);}

}
