#ifndef LIB_NN_SERIALISABLE_HPP_
#define LIB_NN_SERIALISABLE_HPP_

#include <string>

class Serialisable {
 public:
  template <class T>
  std::string serialise() {
    return std::string((char*)this, (char*)(this + sizeof(T)));
  }
  template <class T>
  static T* deserialise(const char* buf) {
    return (T*)buf;
  }
};

#endif  // LIB_NN_SERIALISABLE_HPP_