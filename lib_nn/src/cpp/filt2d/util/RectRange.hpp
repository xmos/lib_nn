#pragma once

#include <vector>
#include <memory>
#include <cassert>


class RectRange {

  public:

    struct Dim { int start, end, step; };

    class iterator: public std::iterator<std::input_iterator_tag, std::vector<int>> {

      private:

        RectRange& parent;
        std::vector<int> val;

      public:

        explicit iterator(RectRange& parent, std::vector<int> val)
          : parent(parent), val(val) {}

        iterator& operator++(){
          for(int i = val.size()-1; i >= 0; --i){
            auto dim = parent.getDim(i);
            
            val[i] += dim.step;

            if(val[i] < dim.end){
              for(int j = i+1; j < val.size(); ++j)
                val[j] = parent.getDim(j).start; // I know this isn't great, but I think it should work..
              break;
            } else {
              val[i] = dim.end;
            }
          }

          return *this;
        }

        iterator operator++(int){ iterator retval = *this; ++(*this); return retval; }
        bool operator==(iterator other) const { return this->val == other.val; }
        bool operator!=(iterator other) const { return this->val != other.val; }
        reference operator*() { return this->val; }

    };

  private:

    std::vector<Dim> dims;

  public:

    RectRange(std::initializer_list<Dim> il)
      : dims(il) {}

    const Dim& getDim(int i){ return this->dims[i]; }

    iterator begin() {
      auto v = std::vector<int>(dims.size());
      for(int i = 0; i < dims.size(); ++i)
        v[i] = dims[i].start;
      return iterator(*this, v);
    }

    iterator end() {
      auto v = std::vector<int>(dims.size());
      for(int i = 0; i < dims.size(); ++i)
        v[i] = dims[i].end;
      return iterator(*this, v);
    }

};