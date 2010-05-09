#ifndef BOOM_NYI_HPP
#define BOOM_NYI_HPP

#include <iostream>

namespace BOOM{

  inline void nyi(const std::string & thing){
    std::cerr << thing << " is not yet implemented.\n";
    exit(0);
  }
}
#endif // BOOM_NYI_HPP
