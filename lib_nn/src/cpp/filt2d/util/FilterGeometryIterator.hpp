#pragma once

#include "../geom/Filter2dGeometry.hpp"

#include <iostream>
#include <functional>

namespace nn {
namespace filt2d {

class FilterGeometryIterator {

  public:

    using FilterGeometry = geom::Filter2dGeometry;

    // I wanted this to be a constexpr, but I can't make it work.
    static FilterGeometry END() { 
      return FilterGeometry(geom::ImageGeometry(0,0,0),
                            geom::ImageGeometry(0,0,0),
                            geom::WindowGeometry(0,0,0,0,0,0,0,0,0,0));
    }

    class iterator: public std::iterator<std::input_iterator_tag, FilterGeometry> {

      private:

        FilterGeometryIterator& parent;
        FilterGeometry filter;

      public:

        explicit iterator(FilterGeometryIterator& parent, FilterGeometry filter)
          : parent(parent), filter(filter) {}

        iterator& operator++(){ this->parent.Next(this->filter); return *this; }

        iterator operator++(int){ iterator retval = *this; ++(*this); return retval; }
        bool operator==(iterator other) const { return this->filter == other.filter; }
        bool operator!=(iterator other) const { return this->filter != other.filter; }
        reference operator*() { return this->filter; }

    };

  protected:

    virtual FilterGeometry First() const {
      return FilterGeometry(
        geom::ImageGeometry(1, 1, 32),
        geom::ImageGeometry(1, 1, 16),
        geom::WindowGeometry(1, 1, 32, 0, 0, 1, 1));
    }

    virtual void Next(FilterGeometry& filter) const = 0;

  public:

    iterator begin() {
      return iterator(*this, this->First());
    }

    iterator end() {
      return iterator(*this, FilterGeometryIterator::END());
    }

};



class PredicateFilterGeometryIterator : public FilterGeometryIterator {

  public:

    using FilterGeometryPredicate = std::function<bool(FilterGeometry&)>;

  protected:

    struct {
      struct {
        unsigned min, max, step;
      } height, width, depth;
    } input, output;

    struct {
      struct {
        struct {
          unsigned min, max, step;
        } height, width, depth;
      } shape;

      struct {
        struct {
          int min, max, step;
        } row, col;
      } start;

      struct {
        struct {
          int min, max, step;
        } row, col, channel;
      } stride;

      struct {
        struct {
          int min, max, step;
        } row, col;
      } dilation;
    } window;

    FilterGeometryPredicate predicate;
    
    virtual bool InnerNext(FilterGeometry& filter) const {
      #define UPDATE(FIELD)  do {                 \
        if(FIELD.step != 0) {                     \
          filter.FIELD += FIELD.step;             \
          if(filter.FIELD <= FIELD.max)           \
            return true;                          \
          filter.FIELD = FIELD.min; } } while(0)

      UPDATE(window.dilation.col);
      UPDATE(window.dilation.row);
      UPDATE(window.stride.channel);
      UPDATE(window.stride.col);
      UPDATE(window.stride.row);
      UPDATE(window.start.col);
      UPDATE(window.start.row);
      UPDATE(window.shape.depth);
      UPDATE(window.shape.width);
      UPDATE(window.shape.height);
      UPDATE(output.depth);
      UPDATE(output.width);
      UPDATE(output.height);
      UPDATE(input.depth);
      UPDATE(input.width);
      UPDATE(input.height);
      
      #undef UPDATE

      // If it made it here, we're done.
      return false;
    }

  public:

    PredicateFilterGeometryIterator(const FilterGeometry min, 
                                    const FilterGeometry max, 
                                    const FilterGeometry step,
                                    FilterGeometryPredicate predicate)
      : input{ {min.input.height, max.input.height, step.input.height },
               {min.input.width,  max.input.width,  step.input.width  },
               {min.input.depth,  max.input.depth,  step.input.depth} }, 
        output{ {min.output.height, max.output.height, step.output.height },
                {min.output.width,  max.output.width,  step.output.width  },
                {min.output.depth,  max.output.depth,  step.output.depth} }, 
        window{ { {min.window.shape.height, max.window.shape.height, step.window.shape.height  },
                  {min.window.shape.width,  max.window.shape.width,  step.window.shape.width   },
                  {min.window.shape.depth,  max.window.shape.depth,  step.window.shape.depth } },
                { {min.window.start.row, max.window.start.row, step.window.start.row },
                  {min.window.start.col, max.window.start.col, step.window.start.col } },
                { {min.window.stride.row,     max.window.stride.row,     step.window.stride.row     },
                  {min.window.stride.col,     max.window.stride.col,     step.window.stride.col     },
                  {min.window.stride.channel, max.window.stride.channel, step.window.stride.channel } },
                { {min.window.dilation.row, max.window.dilation.row, step.window.dilation.row},
                  {min.window.dilation.col, max.window.dilation.col, step.window.dilation.col} } },
        predicate(predicate) {}
      


    virtual void Next(FilterGeometry& filter) const override {
      do {
        if(!InnerNext(filter)){
          filter = END();
          return;
        }
      } while(!predicate(filter));
    }
}; 


class IOWFilterGeometryIterator : public FilterGeometryIterator {

  protected:

    virtual bool UpdateInput(FilterGeometry& filter) const = 0;
    virtual bool UpdateOutput(FilterGeometry& filter) const = 0;
    virtual bool UpdateWindow(FilterGeometry& filter) const = 0;

  public:

    virtual void Next(FilterGeometry& filter) const override {

      if(UpdateWindow(filter))
        if(UpdateOutput(filter))
          if(UpdateInput(filter))
            filter = END();

    }

}; 



// class TypicalFilterGeometryIterator : public IOWFilterGeometryIterator {

//   protected:

//     bool is_depthwise;
//     bool supports_padding;
//     bool allows_dilation;
//     unsigned channels_per_cog;

//     struct {
//       struct {
//         unsigned min, max, step;
//       } height, width, depth;
//     } input, output;

//   public:

//     virtual bool UpdateInput(FilterGeometry& filter) const override {
//       // Output and Window will already have been set to their minimums, but they can be adjusted here if need be

//       filter.input.depth += input.depth.step;
//       if(filter.input.depth <= input.depth.max)
//         return false;
//       filter.input.depth = input.depth.min;

//       filter.input.width += input.width.step;
//       if(filter.input.width <= input.width.max)
//         return false;
//       filter.input.width = input.width.min;
      
//       filter.input.height += input.height.step;
//       if(filter.input.height <= input.height.max)
//         return false;
      
//       return true;
//     }

//     virtual bool UpdateOutput(FilterGeometry& filter) const override {
//       // Window will already have been set to its minimums, but they can be adjusted here if need be
      
//       if(is_depthwise){
//         // If the filter is depthwise, this must be the same as the input depth
//         filter.output.depth = filter.input.depth;
//       } else {
//         // Otherwise, it works like input depth
//         filter.output.depth += input.output.step;
//         if(filter.output.depth <= input.depth.max)
//           return false;
//         filter.output.depth = input.output.min;
//       }

//       // Output width must not be so large that the convolution window would entirely leave the input image, so it is
//       // constrained by the 
//       filter.output.width += output.width.step;
//       if(filter.output.width <= output.width.max)
//         return false;
//       filter.output.width = output.width.min;
      
//       filter.output.height += output.height.step;
//       if(filter.output.height <= output.height.max)
//         return false;
      
//       return true;
//     }

//     virtual bool UpdateWindow(FilterGeometry& filter) const override {
//       return true;
//     }
// };



}}