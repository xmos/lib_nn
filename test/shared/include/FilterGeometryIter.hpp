#pragma once

#include <memory>
#include <vector>

#include "geom/Filter2dGeometry.hpp"

namespace nn {
namespace ff {

/**
 *
 */
class IFilterFrame {
 public:
  virtual const nn::Filter2dGeometry& Filter() = 0;
  virtual void Push(const nn::Filter2dGeometry& filter) {}
  virtual bool UpdateFilter() = 0;
};

/**
 *
 */
class FilterFrame : public IFilterFrame {
 protected:
  nn::Filter2dGeometry filter;

 public:
  virtual const nn::Filter2dGeometry& Filter() override { return filter; }
};

/**
 *
 */
class FrameStack : public IFilterFrame {
 private:
  std::vector<std::shared_ptr<IFilterFrame>> frames;

 public:
  FrameStack() {}
  FrameStack(std::initializer_list<IFilterFrame*> filter_frames);
  virtual const nn::Filter2dGeometry& Filter() override;
  virtual void Push(const nn::Filter2dGeometry& filter) override;
  virtual bool UpdateFilter() override;
  bool operator==(FrameStack& other) const;
  bool operator!=(FrameStack& other) const;
};

/**
 *
 */
class Apply : public FilterFrame {
 public:
  using BindFunc = std::function<nn::Filter2dGeometry(nn::Filter2dGeometry)>;

 private:
  BindFunc func;

 public:
  Apply(BindFunc func);
  virtual bool UpdateFilter() override { return false; }
  virtual void Push(const nn::Filter2dGeometry& filter) override;
};

/**
 *
 */
class FrameSequence : public IFilterFrame {
 private:
  std::vector<std::shared_ptr<IFilterFrame>> frames;
  int current_index;

 protected:
  std::shared_ptr<IFilterFrame> Current() { return frames[current_index]; }

 public:
  FrameSequence(std::initializer_list<IFilterFrame*> filter_frames);
  virtual const nn::Filter2dGeometry& Filter() override;
  virtual void Push(const nn::Filter2dGeometry& filter) override;
  virtual bool UpdateFilter() override;
};

/**
 *
 */
class AllUnpadded : public FilterFrame {
 private:
  nn::Filter2dGeometry max_geometry;
  int channel_step;
  bool depthwise;

 protected:
  virtual void Push(const nn::Filter2dGeometry& new_filter) override;
  bool _UpdateFilter();
  virtual bool UpdateFilter() override;

 public:
  AllUnpadded(const nn::Filter2dGeometry max_geometry,
              const bool depthwise = false, const int channel_step = 4);
};

/**
 *
 */
class AllPaddedBase : public FilterFrame {
 private:
  nn::Filter2dGeometry max_geometry;
  int channel_step;
  bool depthwise;

 protected:
  virtual void Push(const nn::Filter2dGeometry& new_filter) override;
  virtual bool UpdateFilter() override;

 public:
  AllPaddedBase(const nn::Filter2dGeometry max_geometry,
                const bool depthwise = false, const int channel_step = 4);
};

/**
 *
 */
class MakePadded : public FilterFrame {
 private:
  nn::padding_t max_padding;
  nn::padding_t cur_padding;
  bool depthwise;

 protected:
  virtual void Push(const nn::Filter2dGeometry& new_filter) override;
  bool _UpdateFilter();
  virtual bool UpdateFilter() override;

 public:
  MakePadded(const nn::padding_t max_padding, const bool depthwise = false);
};

/**
 *
 */
class IntDimensionRange : public FilterFrame {
 private:
  int min;
  int max;
  int step;

 protected:
  virtual void Push(const nn::Filter2dGeometry& new_filter) override;
  virtual bool UpdateFilter() override;
  virtual int& Field() = 0;

 public:
  IntDimensionRange(int min, int max, int step);
};

/**
 *
 */
template <int nn::ImageGeometry::*T_dim>
class ImageDimension : public IntDimensionRange {
 protected:
  virtual nn::ImageGeometry& Image() = 0;
  virtual int& Field() override { return Image().*T_dim; }

 public:
  ImageDimension(int min, int max, int step = 1)
      : IntDimensionRange(min, max, step) {}
};

/**
 *
 */
template <int nn::ImageGeometry::*T_dim>
class InputImageDimension : public ImageDimension<T_dim> {
 protected:
  virtual nn::ImageGeometry& Image() override { return this->filter.input; }

 public:
  InputImageDimension(int min, int max, int step = 1)
      : ImageDimension<T_dim>(min, max, step) {}
};

/**
 *
 */
template <int nn::ImageGeometry::*T_dim>
class OutputImageDimension : public ImageDimension<T_dim> {
 protected:
  virtual nn::ImageGeometry& Image() override { return this->filter.output; }

 public:
  OutputImageDimension(int min, int max, int step = 1)
      : ImageDimension<T_dim>(min, max, step) {}
};

/**
 *
 */
template <int nn::ImageGeometry::*T_dim>
class WindowShapeDimension : public ImageDimension<T_dim> {
 protected:
  virtual nn::ImageGeometry& Image() override {
    return this->filter.window.shape;
  }

 public:
  WindowShapeDimension(int min, int max, int step = 1)
      : ImageDimension<T_dim>(min, max, step) {}
};

using InputHeight = InputImageDimension<&ImageGeometry::height>;
using InputWidth = InputImageDimension<&ImageGeometry::width>;
using InputDepth = InputImageDimension<&ImageGeometry::depth>;

using OutputHeight = OutputImageDimension<&ImageGeometry::height>;
using OutputWidth = OutputImageDimension<&ImageGeometry::width>;
using OutputDepth = OutputImageDimension<&ImageGeometry::depth>;

using WindowHeight = WindowShapeDimension<&ImageGeometry::height>;
using WindowWidth = WindowShapeDimension<&ImageGeometry::width>;

/**
 *
 */
class InputShape : public FrameStack {
 public:
  InputShape(const std::array<int, 3> min, const std::array<int, 3> max,
             const std::array<int, 3> step)
      : FrameStack({new InputHeight(min[0], max[0], step[0]),
                    new InputWidth(min[1], max[1], step[1]),
                    new InputDepth(min[2], max[2], step[2])}) {}
};

/**
 *
 */
class OutputShape : public FrameStack {
 public:
  OutputShape(const std::array<int, 3> min, const std::array<int, 3> max,
              const std::array<int, 3> step)
      : FrameStack({new OutputHeight(min[0], max[0], step[0]),
                    new OutputWidth(min[1], max[1], step[1]),
                    new OutputDepth(min[2], max[2], step[2])}) {}
};

/**
 *
 */
class WindowShape : public FrameStack {
 public:
  WindowShape(const std::array<int, 2> min, const std::array<int, 2> max,
              const std::array<int, 2> step)
      : FrameStack({new WindowHeight(min[0], max[0], step[0]),
                    new WindowWidth(min[1], max[1], step[1])}) {}
};

/**
 *
 */
class FilterGeometryIterator
    : public std::iterator<std::input_iterator_tag, nn::Filter2dGeometry> {
 private:
  FrameStack frames;
  nn::Filter2dGeometry seed;

 public:
  FilterGeometryIterator(nn::Filter2dGeometry seed,
                         std::initializer_list<IFilterFrame*> filter_frames);
  FilterGeometryIterator(std::initializer_list<IFilterFrame*> filter_frames);

  FilterGeometryIterator& operator++();

  bool operator==(FilterGeometryIterator& other) const;
  bool operator!=(FilterGeometryIterator& other) const;
  const nn::Filter2dGeometry& operator*();

  void Reset();

  /**
   * Get the initial iterator
   */
  FilterGeometryIterator& begin();

  /**
   * Get the final iterator. Will always be FilterGeometryIterBase::END.
   */
  const FilterGeometryIterator& end();
};

nn::Filter2dGeometry MakeUnpaddedDepthwise(nn::Filter2dGeometry);
nn::Filter2dGeometry MakePaddedDepthwise(nn::Filter2dGeometry);

nn::Filter2dGeometry MakeUnpaddedDense(nn::Filter2dGeometry);
nn::Filter2dGeometry MakePaddedDense(nn::Filter2dGeometry);

}  // namespace ff
}  // namespace nn