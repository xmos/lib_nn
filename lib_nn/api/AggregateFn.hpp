#pragma once

#include <array>
#include <vector>

#include "Utils.hpp"
#include "xs3_vpu.h"
#include "vpu.hpp"

#include "geom/Filter2dGeometry.hpp"

namespace nn
{

   class AggregateFn
   {
   public:
      virtual void aggregate_fn(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group) = 0;
   };

   struct Conv2dReorderedWeights
   {
      std::vector<int8_t> weights;
      std::vector<int> final_vpu_load_addresses;
      Conv2dReorderedWeights(int channels) : weights(),
                                             final_vpu_load_addresses(channels, 0)
      {
      }
   };

   class MatMulInt8 : public AggregateFn
   {

   public:
      class Params
      {
      public:
         const int8_t *weights;
         const int32_t output_slice_channel_count;
         const int32_t bytes_per_kernel_channel;

         Params(const int output_slice_channel_count, const int32_t bytes_per_kernel_channel, const int8_t *weights);
      };
      Params *params;

   private:
      void mat_mul_impl(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group);

   public:
      MatMulInt8(Params *params) : params(params){};
      void aggregate_fn(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group);

      static Conv2dReorderedWeights reorder_kernel_weights(
          int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element, int8_t pad_value);

      static int get_kernel_size(int input_bytes, int output_channel_count);
      static int get_scratch_size(int input_bytes);
   };

   class MatMulDirectFn : public AggregateFn
   {

   public:
      class Params
      {
      public:
         const int8_t *weights;

         //This has been scaled by VPU_INT16_EPV (bytes_per_kernel_channel_group?)
         int32_t bytes_per_kernel_channel;

         int32_t k_height_loop_counter;
         int32_t k_width_loop_counter;
         int32_t input_channel_loop_counter;

         int32_t inner_x_h_step;
         int32_t inner_x_v_step;

         Params(const ImageGeometry &X, const WindowGeometry &K, const int input_ch_per_output, const int8_t *weights);
      };

   protected:
      void mat_mul_direct_impl(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group);

      Params *params;

   public:
      MatMulDirectFn(Params *params) : params(params){};
      void aggregate_fn(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group);
   };

   class MatMulBinaryDirectFn : public MatMulDirectFn
   {
   public:
      MatMulBinaryDirectFn(Params *params) : MatMulDirectFn(params) {}

   private:
      void mat_mul_direct_impl(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group);
   };

   /**
   * Aggregator for performing maxpool on a contiguous sequence of 32-channel pixels.
   * 
   * @see DirectWriteOutputTransform
   */
   class MaxPoolPatchFn : public AggregateFn,
                          public ChannelParallelComponent<VPU_INT8_EPV_LOG2>
   {

   public:
      /**
       * Configuration parameters for MaxPoolPatchFn
       */
      struct Params
      {

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

         /**
         * Construct a MaxPoolPatchFn::Params by deserializing it from the provided byte stream.
         */
         Params(std::istream &stream);

         /**
         * Serialize this MaxPoolPatchFn::Params into the provided byte stream
         */
         void Serialize(std::ostream &stream) const;
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
      virtual void aggregate_fn(vpu_ring_buffer_t *acc,
                                int8_t *input_patch,
                                int32_t output_channel_group) override;
   };

   /**
   * Parameter struct required by maxpool_direct_valid_ref() and maxpool_direct_valid_xcore().
   */
   typedef struct
   {
      /**
     * Stride between columns in the pooling window (taking dilation into account).
     */
      int32_t col_stride;

      /**
     * The number of columns in the pooling widow
     */
      int32_t cols;

      /**
     * Stride from the last pixel in one row of the pooling window, to the first pixel of the next.
     */
      int32_t row_stride;

      /**
     * The number of rows in the pooling window
     */
      int32_t rows;
   } maxpool_direct_valid_params;

   /**
   * Aggregator for performing maxpool over a 2D rectangular grid of pixels, not necessarily contiguous in memory.
   * 
   * Maxpooling will be applied across 32 channels.
   * 
   * @see DirectWriteOutputTransform
   */
   class MaxPoolDirectValidFn : public AggregateFn,
                                public ChannelParallelComponent<VPU_INT8_EPV_LOG2>
   {

   public:
      /**
       * Configuration parameters for MaxPoolDirectValidFn
       */
      struct Params
      {

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

         /**
         * Construct a MaxPoolDirectValidFn::Params by deserializing it from the provided byte stream
         */
         Params(std::istream &stream);

         /**
         * Seralize this MaxPoolDirectValidFn::Params into the provided byte stream
         */
         void Serialize(std::ostream &stream) const;
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
      virtual void aggregate_fn(vpu_ring_buffer_t *acc,
                                int8_t *input_img,
                                int32_t output_channel_group) override;
   };

   /**
   * xcore implementation of maxpool_patch_ref()
   */
   C_API
   void maxpool_patch_xcore(
       vpu_ring_buffer_t *A, // This doesn't really make sense for maxpool.
       const int8_t *patch,
       const int pixels);

   /**
   * xcore implementation of maxpool_direct_valid_ref()
   */
   C_API
   void maxpool_direct_valid_xcore(
       vpu_ring_buffer_t *A,
       const int8_t *X,
       const maxpool_direct_valid_params *params);

   /**
   * Portable implementation of maxpool_patch_xcore().
   */
   C_API
   void maxpool_patch_ref(
       vpu_ring_buffer_t *A, // This doesn't really make sense for maxpool.
       const int8_t *patch,
       const int pixels);

   /**
   * Portable implementation of maxpool_direct_valid_xcore().
   */
   C_API
   void maxpool_direct_valid_ref(
       vpu_ring_buffer_t *A,
       const int8_t *X,
       const maxpool_direct_valid_params *params);

   /**
   * Parameter struct required by avgpool_patch_ref() and avgpool_patch_xcore().
   */
   typedef struct
   {
      /**
     * Number of pixels in a patch.
     */
      int32_t pixels;

      /**
     * Scale value by which input elements are multiplied.
     * 
     * Each scale[k] == scale[j]. It is duplicated for the VPU's sake. 
     */
      int8_t scale[VPU_INT8_ACC_PERIOD];
   } avgpool_patch_params;

   /**
   * Aggregator for performing avgpool on a contiguous sequence of 16-channel pixels.
   * 
   * @see 
   */
   class AvgPoolPatchFn : public AggregateFn,
                          public ChannelParallelComponent<VPU_INT8_ACC_PERIOD_LOG2>
   {

   public:
      /**
       * Configuration parameters for AvgPoolPatchFn
       */
      struct Params
      {

         /**
         * Parameter struct required by the low-level implementation.
         */
         avgpool_patch_params ap_params;

         /**
        */
         Params() {}

         /**
         * Construct an AvgPoolPatchFn::Params using the supplied params struct.
         */
         Params(const avgpool_patch_params &ap_params);

         /**
         * Construct an AvgPoolPatchFn::Params from a high-level geometric description of the
         * pooling window.
         */
         Params(const nn::WindowGeometry &filter,
                const int8_t scale);

         /**
         * Deserialize an AvgPoolPatchFn::Params from a byte stream.
         */
         Params(std::istream &stream);

         /**
         * Serialize an AvgPoolPatchFn::Params into a byte stream.
         */
         void Serialize(std::ostream &stream) const;
      };

   protected:
      /**
       * The configuration parameters for this operator.
       */
      const Params *params;

   public:
      /**
       * Construct an AvgPoolPatchFn
       */
      AvgPoolPatchFn(const Params *params);

      /**
       * Perform avgpool aggregation.
       */
      virtual void aggregate_fn(vpu_ring_buffer_t *acc,
                                int8_t *input_patch,
                                int32_t output_channel_group) override;
   };

   /**
   * Parameter struct required by avgpool_direct_valid_ref() and avgpool_direct_valid_xcore().
   */
   typedef struct
   {

      /**
     * Stride between columns in the pooling window (taking dilation into account).
     */
      int32_t col_stride;

      /**
     * The number of columns in the pooling widow
     */
      int32_t cols;

      /**
     * Stride from the last pixel in one row of the pooling window, to the first pixel of the next.
     */
      int32_t row_stride;

      /**
     * The number of rows in the pooling window
     */
      int32_t rows;

      /**
     * Scale value by which input elements are multiplied.
     * 
     * Each scale[k] == scale[j]. It is duplicated for the VPU's sake. 
     */
      int8_t scale[VPU_INT8_ACC_PERIOD];
   } avgpool_direct_valid_params;

   /**
   * Aggregator for performing avgpool over a 2D rectangular grid of pixels, not necessarily contiguous in memory.
   * 
   * AvgPoolDirectValidFn can process up to 16 output channels in parallel.
   * 
   * @see ShiftInt8OutputTransform
   */
   class AvgPoolDirectValidFn : public AggregateFn,
                                public ChannelParallelComponent<VPU_INT8_ACC_PERIOD_LOG2>
   {

   public:
      /**
       * Configuration parameters for AvgPoolDirectValidFn
       */
      struct Params
      {

         /**
         * Parameters to be passed to the kernel function.
         */
         avgpool_direct_valid_params ap_params;

         /**
         * 
         */
         Params() {}

         /**
         * Construct a AvgPoolDirectValidFn::Params
         */
         Params(const avgpool_direct_valid_params &ap_params);

         /**
         * Construct a AvgPoolDirectValidFn::Params
         */
         Params(const nn::Filter2dGeometry &filter,
                const int8_t scale);

         /**
         * Deserialize an AvgPoolDirectValidFn::Params from a byte stream.
         */
         Params(std::istream &stream);

         /**
         * Serialize an AvgPoolDirectValidFn::Params into a byte stream.
         */
         void Serialize(std::ostream &stream) const;
      };

   protected:
      /**
       * The configuration parameters for this operator.
       */
      const Params *params;

   public:
      /**
       * Construct an AvgPoolDirectValidFn
       */
      AvgPoolDirectValidFn(const Params *params);

      /**
       * Perform avgpool aggregation.
       */
      virtual void aggregate_fn(vpu_ring_buffer_t *acc,
                                int8_t *input_img,
                                int32_t output_channel_group) override;
   };

   /**
   * xcore implementation of avgpool_patch_ref()
   */
   C_API
   void avgpool_patch_xcore(
       vpu_ring_buffer_t *A,
       const int8_t patch[],
       const avgpool_patch_params *params);

   /**
   * xcore implementation of avgpool_direct_valid_ref()
   */
   C_API
   void avgpool_direct_valid_xcore(
       vpu_ring_buffer_t *acc,
       const int8_t X[],
       const avgpool_direct_valid_params *params);

   /**
   * Portable implementation of avgpool_patch_xcore().
   */
   C_API
   void avgpool_patch_ref(
       vpu_ring_buffer_t *A,
       const int8_t patch[],
       const avgpool_patch_params *params);

   /**
   * Portable implementation of avgpool_direct_valid_xcore().
   */
   C_API
   void avgpool_direct_valid_ref(
       vpu_ring_buffer_t *acc,
       const int8_t X[],
       const avgpool_direct_valid_params *params);
}
