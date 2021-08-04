#ifndef LIB_NN_SERIALISABLE_HPP_
#define LIB_NN_SERIALISABLE_HPP_

#include <string>

class Serialisable {
 public:

  /**
   * @brief Serialise a class to a string, if the class contains any pointers then 
   * the serialisation and deserialisation needs to be overridden.
   * 
   * @tparam T The derived class.
   * @return std::string 
   */
  template <class T>
  std::string serialise() {
    return std::string((char*)this, (char*)(this + sizeof(T)));
  }

  /**
   * @brief Get the variable allocation byte count. The number of bytes to allocate
   * will be all the constant size objects + this much.
   * 
   * @param input_buffer 
   * @return int 
   */
  int get_variable_allocation_byte_count(const char* input_buffer){
    return 0;
  }
  
  /**
   * @brief Deserialise the input_buffer into the allocated_memory.
   * 
   * @tparam T 
   * @param allocated_memory 
   * @param input_buffer 
   * @return T* 
   */
  template <class T>
  static T* deserialise(char* allocated_memory, const char* input_buffer) {
    std::memcpy(allocated_memory, input_buffer , sizeof(T));
    return (T*)allocated_memory;
  }
};

#endif  // LIB_NN_SERIALISABLE_HPP_