#ifndef LIB_NN_AGGREGATE_FN_HPP_
#define LIB_NN_AGGREGATE_FN_HPP_

#include <array>
#include <vector>

#include "Serialisable.hpp"
#include "Utils.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "vpu.hpp"
#include "xs3_vpu.h"

namespace nn {

class AggregateFn {
 public:
  /**
   * @brief Function to aggregate an input receptive field as defined by the
   * operation it is performing, i.e. convolution, maximum, addition, etc.
   * Aggregation is performed in batches of channel groups, each of the same
   * size, with the final channel group possibly being of fewer output channels.
   * A channel group is defined by the VPU ring buffer length of 16.
   *
   * @param A Pointer to enough memory to hold the VPU ring buffer.
   * @param T Pointer to enough scratch memory to perform the aggregate
   * function.
   * @param output_channel_group Denotes which channel group will be computed.
   */
  virtual void aggregate_fn(VPURingBuffer *A, int8_t *T,
                            int32_t output_channel_group) = 0;
};

struct Conv2dReorderedWeights {
  /**
   * @brief Byte vector of the weights in the order that the VPU can most
   * efficiently execute.
   *
   */
  std::vector<int8_t> weights;

  /**
   * @brief vector of offsets into the weights. Each of the offsets represents
   * the location that the VPU will load from in during the final load. The
   * purpose of this vector is to allow the computation of an accumulator
   * adjustment where nessessary. For example the final load might look like: |
   * N bytes of weights | 32 - N bytes of unwanted weights | where N is between
   * 1 and 32. Knowing the 32 - N bytes of unwanted weights coupled with the
   * scratch memory padding value allows the computation of a statically
   * knowable accumulator offset. This offset can then be included into the
   * bias. This allows dense packing of weights.
   */
  std::vector<int> final_vpu_load_addresses;

  Conv2dReorderedWeights(int channels)
      : weights(), final_vpu_load_addresses(channels, 0) {}
};

class MatMulBase : public AggregateFn {
 public:
  class Params : public Serialisable {
   public:
    int32_t output_slice_channel_count;
    int32_t bytes_per_kernel_channel;
    /**
     * @brief Construct a new Params object.
     *
     * @param output_slice_channel_count The count of output channels to be
     * computed by this parameter set.
     * @param bytes_per_kernel_channel The count of bytes per each channel of
     * the kernel (weights).
     */
    Params(int output_slice_channel_count, int32_t bytes_per_kernel_channel);
  };

 protected:
  /**
   * @brief This describes the region over which this class will perform its
   * operation(MatMul).
   */
  Params *params;
  int8_t *weights;

 public:
  MatMulBase(Params *params) : params(params){};

  /**
   * Setter for the reordered weights
   */
  void setWeights(int8_t *reordered_weights) { weights = reordered_weights; }

  /**
   * @brief Used to reorder the weights from their normal form ([OutputChannel,
   * Height, Width, InputChannel]) to a form conducive to the VPU efficiently
   * loading and multiplying them.
   *
   * @param raw_weights Pointer to the raw weights.
   * @param shape [OutputChannels, Height, Width, InputChannels]
   * @param bits_per_element The count of bits per element, i.e. int8 is 8,
   * binary is 1.
   * @param pad_value The value used for padding to achieve alignment where
   * nessessary. This is not window padding. It can effect the accumulator
   * result and must be compensated for where nessessary, i.e. padding with zero
   * will not effect an int8, int16 or int32 matmul, padding with
   * zeros(representing 1) will effect a binary matmul.
   * @return Conv2dReorderedWeights
   */
  static Conv2dReorderedWeights reorder_kernel_weights(
      int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element,
      int8_t pad_value);

  /**
   * @brief Get the required size of the weights array. This is a non-trivial
   * computation as it accounts for how the VPU will access the weights array.
   * It includes padding at after the weights, this padding exists to ensure
   * that all memory accesses of the VPU are of valid memory.
   * [asj] we could stop emiting this as OOB accesses shouldnt cause a problem
   * and only cost memory.
   *
   * @param input_bytes_per_channel The count of bytes in a single channel, i.e.
   * the product of kernel height, kernel width, input channel count and bytes
   * per element.
   * @param output_channel_count The number of output channels that these
   * weights will compute.
   * @return The size in bytes that will hold the weights and guarentee no OOB
   * accesses.
   */
  static int get_weights_bytes(int input_bytes_per_channel,
                               int output_channel_count);

  /**
   * @brief Get the required size of the scratch array. This is a non-trivial
   * computation as it accounts for how the VPU will access the scratch array.
   *
   * @param input_bytes The count of bytes in a single channel.
   * @return The size in bytes that will hold the copy of the current patch and
   * guarentee no OOB accesses.
   */
  static int get_scratch_mem_bytes(int input_bytes);
};

class MatMulInt8 : public MatMulBase {
 public:
  MatMulInt8(MatMulBase::Params *params) : MatMulBase(params){};
  void aggregate_fn(VPURingBuffer *A, int8_t *T, int32_t output_channel_group);
};

class MatMulBinary : public MatMulBase {
 public:
  MatMulBinary(MatMulBase::Params *params) : MatMulBase(params){};
  void aggregate_fn(VPURingBuffer *A, int8_t *T, int32_t output_channel_group);
};

class MatMulDirectFn : public AggregateFn {
 public:
  class Params : public Serialisable {
   public:
    // This has been scaled by VPU_INT16_EPV (bytes_per_kernel_channel_group?)
    int32_t bytes_per_kernel_channel;

    int32_t k_height_loop_counter;
    int32_t k_width_loop_counter;
    int32_t input_channel_loop_counter;

    int32_t inner_x_h_step;
    int32_t inner_x_v_step;

    /**
     * @brief Construct a new Params object
     *
     * @param X Class describing the properties of the input the convolution
     * will be performed over.
     * @param K Class describing the properties of the convolution to be
     * preformed.
     * @param input_ch_per_output The count of input channeks that contribute to
     * an output channel. For example, a depthwise convolution will have one
     * input channel per output channel whereas a Conv2D will most likely have
     * the input tensors channel count.
     */
    Params(const ImageGeometry &X, const WindowGeometry &K,
           const int input_ch_per_output);
  };

 protected:
  /**
   * @brief This describes the region over which this class will perform its
   * operation(MatMul).
   */
  Params *params;

  /**
   * @brief A Pointer to the begining of the reordered weights.
   */
  int8_t *weights;

 public:
  MatMulDirectFn(Params *params) : params(params){};

  void aggregate_fn(VPURingBuffer *A, int8_t *T, int32_t output_channel_group);

  /**
   * Setter for the reordered weights
   */
  void setWeights(int8_t *reordered_weights) { weights = reordered_weights; }
};

class MatMulBinaryDirectFn : public MatMulDirectFn {
 public:
  MatMulBinaryDirectFn(Params *params) : MatMulDirectFn(params){};
  void aggregate_fn(VPURingBuffer *A, int8_t *T, int32_t output_channel_group);
};

// Depthwise below here
// ////////////////////////////////////////////////////////////////////////////

class MatMulDirectFn_DW : public AggregateFn {
 public:
  class Params : public Serialisable {
   public:
    /*
     The count of bytes that a channel group contains. It is used to dereference
     the weights pointer to the correct channel group start, i.e.
      int8_t *K_p = params->weights + bytes_per_kernel_channel_group *
     output_channel_group;
    */
    int32_t bytes_per_kernel_channel_group;

    int32_t k_height_loop_counter;
    int32_t k_width_loop_counter;

    int32_t inner_x_h_step;
    int32_t inner_x_v_step;

    /**
     * @brief Construct a new Params object for direct application to the input
     * tensor
     *
     * @param X Class describing the properties of the input the convolution
     * will be performed over.
     * @param K Class describing the properties of the convolution to be
     * preformed.
     * @param weights A Pointer to the begining of the reordered weights.
     * @param weights_bytes Count of bytes in the weights array.
     */
    Params(const ImageGeometry &X, const WindowGeometry &K);

    /**
     * @brief Construct a new Params object for indirect application to the
     * input tensor via a scratch array.
     *
     * @param K Class describing the properties of the convolution to be
     * preformed.
     * @param weights A Pointer to the begining of the reordered weights.
     */
    Params(const WindowGeometry &K);
  };

 protected:
  /**
   * @brief This describes the region over which this class will perform its
   * operation(MatMul).
   */
  Params *params;
  int8_t *weights;

 public:
  MatMulDirectFn_DW(Params *params) : params(params){};

  void aggregate_fn(VPURingBuffer *A, int8_t *T, int32_t output_channel_group);

  /**
   * Setter for the reordered weights
   */
  void setWeights(int8_t *reordered_weights) { weights = reordered_weights; }
  /**
   * @brief Used to reorder the weights from their normal form ([OutputChannel,
   * Height, Width, InputChannel]) to a form conducive to the VPU efficiently
   * loading and multiplying them.
   *
   * @param raw_weights Pointer to the raw weights.
   * @param shape [OutputChannels, Height, Width, InputChannels]
   * @param bits_per_element The count of bits per element, i.e. int8 is 8,
   * binary is 1.
   * @param pad_value The value used for padding to achieve alignment where
   * nessessary. This is not window padding. It can effect the accumulator
   * result and must be compensated for where nessessary, i.e. padding with zero
   * will not effect an int8, int16 or int32 matmul, padding with
   * zeros(representing 1) will effect a binary matmul.
   * @return Conv2dReorderedWeights
   */
  static Conv2dReorderedWeights reorder_kernel_weights(
      int8_t *raw_weights, std::array<int, 4> &shape, int8_t pad_value);

  /**
   * @brief Get the required size of the weights array. This is a non-trivial
   * computation as it accounts for how the VPU will access the weights array.
   *
   * @param input_bytes The number of bytes a single output channel of the
   kernel requires, i.e. kernel height x kernel width.
   * @param output_channel_count The number of output channels that these
   * weights will compute.
   * @return The size in bytes that will hold the weights and guarentee no OOB
   * accesses.
   */
  static int get_weights_bytes(int input_bytes, int output_channel_count);

  /**
   * @brief Get the required size of the scratch array. This is a non-trivial
   * computation as it accounts for how the VPU will access the scratch array.
   *
   * @param kernel_shape The shape of the kernel. It should look like: [1,
   * k_height, k_width, input channels]
   * @return The size in bytes that will hold the copy of the current patch and
   * guarentee no OOB accesses.
   */
  static int get_scratch_mem_bytes(std::array<int, 4> &kernel_shape);
};

/**
 * Aggregator for performing maxpool on a contiguous sequence of 32-channel
 * pixels.
 *
 * @see DirectWriteOutputTransform
 */
class MaxPoolPatchFn : public AggregateFn,
                       public ChannelParallelComponent<VPU_INT8_EPV_LOG2> {
 public:
  /**
   * Configuration parameters for MaxPoolPatchFn
   */
  struct Params {
    /**
     * The number of pixels in a patch.
     */
    int32_t pixel_count;

    /**
     * Construct a MaxPoolPatchFn::Params
     */
    Params(const int32_t pixel_count);

    /**
     * Construct a MaxPoolPatchFn::Params
     */
    Params(const nn::WindowGeometry &window);
  };

 protected:
  /**
   * Configuration parameters for this MaxPoolPatchFn
   */
  const Params *params;

 public:
  /**
   * Construct a MaxPoolPatchFn
   */
  MaxPoolPatchFn(const Params *params);

  /**
   * Perform maxpool aggregation.
   */
  virtual void aggregate_fn(VPURingBuffer *acc, int8_t *input_patch,
                            int32_t output_channel_group) override;
};

/**
 * Parameter struct required by maxpool_direct_valid_ref() and
 * maxpool_direct_valid_xcore().
 */
typedef struct {
  /**
   * Stride between columns in the pooling window (taking dilation into
   * account).
   */
  int32_t col_stride;

  /**
   * The number of columns in the pooling widow
   */
  int32_t cols;

  /**
   * Stride from the last pixel in one row of the pooling window, to the first
   * pixel of the next.
   */
  int32_t row_stride;

  /**
   * The number of rows in the pooling window
   */
  int32_t rows;
} maxpool_direct_valid_params;

/**
 * Aggregator for performing maxpool over a 2D rectangular grid of pixels, not
 * necessarily contiguous in memory.
 *
 * Maxpooling will be applied across 32 channels.
 *
 * @see DirectWriteOutputTransform
 */
class MaxPoolDirectValidFn
    : public AggregateFn,
      public ChannelParallelComponent<VPU_INT8_EPV_LOG2> {
 public:
  /**
   * Configuration parameters for MaxPoolDirectValidFn
   */
  struct Params {
    /**
     * Parameters to be passed to the kernel function.
     */
    maxpool_direct_valid_params mp_params;

    /**
     * Construct a MaxPoolDirectValidFn::Params
     */
    Params(const maxpool_direct_valid_params &mp_params);

    /**
     * Construct a MaxPoolDirectValidFn::Params
     */
    Params(const nn::ImageGeometry &input_img,
           const nn::WindowGeometry &window);
  };

 protected:
  /**
   * Configuration parameters for this MaxPoolDirectValidFn.
   */
  const Params *params;

 public:
  /**
   * Construct a MaxPoolDirectValidFn
   */
  MaxPoolDirectValidFn(const Params *params);

  /**
   * Perform maxpool aggregation.
   */
  virtual void aggregate_fn(VPURingBuffer *acc, int8_t *input_img,
                            int32_t output_channel_group) override;
};

/**
 * xcore implementation of maxpool_patch_ref()
 */
C_API
void maxpool_patch_xcore(
    VPURingBuffer *A,  // This doesn't really make sense for maxpool.
    const int8_t *patch, const int pixels);

/**
 * xcore implementation of maxpool_direct_valid_ref()
 */
C_API
void maxpool_direct_valid_xcore(VPURingBuffer *A, const int8_t *X,
                                const maxpool_direct_valid_params *params);

/**
 * Portable implementation of maxpool_patch_xcore().
 */
C_API
void maxpool_patch_ref(
    VPURingBuffer *A,  // This doesn't really make sense for maxpool.
    const int8_t *patch, const int pixels);

/**
 * Portable implementation of maxpool_direct_valid_xcore().
 */
C_API
void maxpool_direct_valid_ref(VPURingBuffer *A, const int8_t *X,
                              const maxpool_direct_valid_params *params);

}  // namespace nn

#endif  // LIB_NN_AGGREGATE_FN_HPP_
