#pragma once

#include <cmath>
#include <limits>

#include "Rand.hpp"
#include "geom/Filter2dGeometry.hpp"

struct SimpleFilter {
  bool depthwise;
  bool padded;
  unsigned cog_size;

  SimpleFilter(bool depthwise = false, bool padded = false,
               unsigned cog_size = 16)
      : depthwise(depthwise), padded(padded), cog_size(cog_size) {}
};

using namespace nn::test;
using namespace nn;

/**
 */
template <>
inline Filter2dGeometry Rand::rand<Filter2dGeometry, SimpleFilter>(
    SimpleFilter p) {
  if (p.padded) {
    const int K_h = this->rand<int>(2, 4);
    const int K_w = this->rand<int>(2, 4);

    const int pad_t = this->rand<int>(1, K_h - 1);
    const int pad_l = this->rand<int>(1, K_w - 1);
    const int pad_b = this->rand<int>(1, K_h - 1);
    const int pad_r = this->rand<int>(1, K_w - 1);

    const int Y_h = this->rand<int>(2, 10);
    const int Y_w = this->rand<int>(2, 10);

    const int X_h = Y_h * K_h - (pad_t + pad_b);
    const int X_w = Y_w * K_w - (pad_l + pad_r);

    const int X_c = this->rand<int>(1, 10) * 4;
    const int Y_c = p.depthwise ? X_c : (this->rand<int>(1, 10) * 4);
    const int W_c = p.depthwise ? 1 : X_c;
    const int W_stride_chan = p.depthwise ? 1 : 0;

    return Filter2dGeometry(
        ImageGeometry(X_h, X_w, X_c), ImageGeometry(Y_h, Y_w, Y_c),
        WindowGeometry(K_h, K_w, W_c, -pad_t, -pad_l, K_h, K_w, W_stride_chan));

  } else {
    const int K_h = this->rand<int>(3, 3);
    const int K_w = this->rand<int>(4, 4);

    const int Y_h = this->rand<int>(1, 10);
    const int Y_w = this->rand<int>(1, 10);

    const int X_h = Y_h * K_h;
    const int X_w = Y_w * K_w;

    const int X_c = this->rand<int>(1, 10) * 4;
    const int Y_c = p.depthwise ? X_c : (this->rand<int>(1, 10) * 4);
    const int W_c = p.depthwise ? 1 : X_c;
    const int W_stride_chan = p.depthwise ? 1 : 0;

    return Filter2dGeometry(
        ImageGeometry(X_h, X_w, X_c), ImageGeometry(Y_h, Y_w, Y_c),
        WindowGeometry(K_h, K_w, W_c, 0, 0, K_h, K_w, W_stride_chan));
  }
}

static inline int8_t round_int8(float val) {
  auto r = std::roundf(val + ldexpf(((val >= 0) ? 1 : -1), -10));
  return int8_t(
      std::max<float>(std::numeric_limits<int8_t>::min(),
                      std::min<float>(std::numeric_limits<int8_t>::max(), r)));
}

static inline std::ostream &operator<<(std::ostream &stream,
                                       const std::vector<int8_t> &v) {
  stream << "[";
  for (int i = 0; i < v.size() - 1; i++) {
    stream << int(v[i]) << ", ";
  }
  if (v.size() != 0) stream << int(v[v.size() - 1]);
  stream << "]";
  return stream;
}

static inline std::ostream &operator<<(std::ostream &stream,
                                       const std::vector<int32_t> &v) {
  stream << "[";
  for (int i = 0; i < v.size() - 1; i++) {
    stream << v[i] << ", ";
  }
  if (v.size() != 0) stream << v[v.size() - 1];
  stream << "]";
  return stream;
}