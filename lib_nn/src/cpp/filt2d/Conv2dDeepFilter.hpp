#pragma once


// #include "Filter2d_util.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "PatchHandlers.hpp"
#include "AggregationHandlers.hpp"
#include "OutputTransformers.hpp"
#include "util/FilterGeometryIterator.hpp"

#include "util/SliceIterator.hpp"

#include <cstdint>


namespace nn {
namespace filt2d {
namespace op {


class Conv2dDeepFilter_Valid
{

  /*********
   * Helpful Type Info
   *********/
  public:

    // Aliases for the concrete types this filter acts upon
    using T_elm_in = int8_t;
    using T_elm_out = int8_t;
    using T_acc = vpu_split_acc32_t;
    using T_coef = int8_t;

    // Aliases for the component classes used by this class
    using T_patch = ValidDeepPatchHandler;
    using T_agg = Conv2dDeepPatchAggregator<T_elm_in, T_coef, T_acc>;
    using T_ot = Int8OutputTransformHandler;

    // Just to make it more readable.
    using FilterGeometry = geom::Filter2dGeometry;
    using InputGeometry = geom::ImageGeometry;
    using OutputGeometry = geom::ImageGeometry;


  /*********
   * Maximum Channel Output Group size 
   *********/

    // Filter processes 16 output channels in parallel
    static constexpr unsigned N_max_cog_chans_log2 = 4;
    static constexpr unsigned N_max_cog_chans = (1<<N_max_cog_chans_log2);

  
  /*********
   * Inner Classes
   *********/
  public:

    /**
     * Represents an individual job, which computes a portion of the output image.
     */
    class Job {

      protected:

        const Conv2dDeepFilter_Valid& filter;

        struct {
          T_patch patch;
          T_agg agg;
          T_ot ot;
        } handler;


      public:

        Job(const Conv2dDeepFilter_Valid& filter, 
            T_patch patch_handler,
            T_agg agg_handler,
            T_ot ot_handler)
            : filter(filter), 
              handler { patch_handler, agg_handler, ot_handler } {}

        void computeSlice(
            const ImageVect& slice_coords, 
            const unsigned out_chan_count);

      protected:

        // const T_elm_in* getInputPointer(const ImageVect& output_coords) const;
        // T_elm_out* getOutputPointer(const ImageVect& output_coords) const;


    };
    
    using JobClass = SliceIterator<Job,SliceType::Channel>;

    /**
     * All configuration info for the filter. Jobs access their shared configuration through this.
     * The patch, agg and ot configs are used when creating handlers for the jobs.
     * 
     * The idea is that this config should be serializable so that it can be easily pulled out of a
     * byte buffer and used.
     */
    struct Config {

        struct { 
          const AddressCovector<T_elm_in> covector; 
        } input;

        struct {
          const AddressCovector<T_elm_out> covector;

          struct {
            const unsigned height;
            const unsigned width;
            const unsigned depth;
          } shape;
        } output;

        struct {
          const T_patch::Config patch;
          const T_agg::Config agg;
          const T_ot::Config ot;
        } handler;

        Config(const InputGeometry& input_geom,
               const OutputGeometry& output_geom,
               const T_patch::Config& patch_config,
               const T_agg::Config& agg_config,
               const T_ot::Config& ot_config)
          : input{input_geom.getAddressCovector()}, 
            output{ output_geom.getAddressCovector(), 
                    {output_geom.height, output_geom.width, output_geom.depth}},
            handler{ patch_config, agg_config, ot_config } {}
        
        Config(const FilterGeometry& filt,
               const T_acc * biases,
               const T_coef * kernel_tensor,
               const nn_acc32_to_int8_params_t* ot_params,
               const bool use_symmetric_saturation = false)
          : Config(filt.input, filt.output, T_patch::Config(filt), 
                   T_agg::Config(biases, kernel_tensor, filt.window),
                   T_ot::Config(ot_params, use_symmetric_saturation)) {}
    };


  /********
   *  Class Methods
   ********/
  public:

    static unsigned inline CogCount(const unsigned channels)
      { return (channels + N_max_cog_chans - 1) >> N_max_cog_chans_log2; }

    static bool SupportsGeometry(const FilterGeometry& filter);

    // Would have made this a constexpr, except the predicate is a std::function,
    // which can't be constexpr'ed
    static nn::filt2d::PredicateFilterGeometryIterator GetGeometryIterator();

  
  /*********
   * Instance member fields
   *********/
  protected:

    const Config config;

    struct {
      const T_elm_in* const input;
      T_elm_out* const output;
    } image;


  public:

    Conv2dDeepFilter_Valid(const Config& config,
                     const T_elm_in* input_image,
                     T_elm_out* output_image)
      : config(config), image{input_image, output_image} {}

    Conv2dDeepFilter_Valid(const T_elm_in* input_image,
                     T_elm_out* output_image,
                     const FilterGeometry& filt,
                     const T_acc * biases,
                     const T_coef * kernel_tensor,
                     const nn_acc32_to_int8_params_t* ot_params,
                     const bool use_symmetric_saturation = false)
      : config(filt, biases, kernel_tensor, ot_params, use_symmetric_saturation), 
        image{input_image, output_image} {}

    void execute(T_elm_in* patch_mem) const;

    JobClass spawnJob(
        const ImageRegion& region,
        T_elm_in* patch_mem) const;

};






class Conv2dDeepFilter
{
  public:

    // Aliases for the concrete types this filter acts upon
    using T_elm_in = int8_t;
    using T_elm_out = int8_t;
    using T_acc = vpu_split_acc32_t;
    using T_coef = int8_t;

    // Aliases for the component classes used by this class
    using T_patch = UniversalPatchHandler;
    using T_agg = Conv2dDeepPatchAggregator<T_elm_in, T_coef, T_acc>;
    using T_ot = Int8OutputTransformHandler;

    // Just to make it more readable.
    using FilterGeometry = geom::Filter2dGeometry;
    using InputGeometry = geom::ImageGeometry;
    using OutputGeometry = geom::ImageGeometry;

    // Filter processes 16 output channels in parallel
    static constexpr unsigned N_max_cog_chans = 16;

  public:

    /**
     * Represents an individual job, which computes a portion of the output image.
     */
    class Job {

      protected:

        const Conv2dDeepFilter& filter;

        struct {
          T_patch patch;
          T_agg agg;
          T_ot ot;
        } handler;


      public:

        Job(const Conv2dDeepFilter& filter, 
            T_patch patch_handler,
            T_agg agg_handler,
            T_ot ot_handler)
            : filter(filter), 
              handler { patch_handler, agg_handler, ot_handler } {}

        void computeSlice(
            const ImageVect& slice_coords, 
            const unsigned out_chan_count);

      protected:

        // const T_elm_in* getInputPointer(const ImageVect& output_coords) const;
        // T_elm_out* getOutputPointer(const ImageVect& output_coords) const;


    };
    
    using JobClass = SliceIterator<Job,SliceType::Channel>;

    /**
     * All configuration info for the filter. Jobs access their shared configuration through this.
     * The patch, agg and ot configs are used when creating handlers for the jobs.
     * 
     * The idea is that this config should be serializable so that it can be easily pulled out of a
     * byte buffer and used.
     */
    struct Config {

        struct { 
          const AddressCovector<T_elm_in> covector; 
        } input;

        struct {
          const AddressCovector<T_elm_out> covector;

          struct {
            const unsigned height;
            const unsigned width;
            const unsigned depth;
          } shape;
        } output;

        struct {
          const T_patch::Config patch;
          const T_agg::Config agg;
          const T_ot::Config ot;
        } handler;

        Config(const InputGeometry& input_geom,
               const OutputGeometry& output_geom,
               const T_patch::Config& patch_config,
               const T_agg::Config& agg_config,
               const T_ot::Config& ot_config)
          : input{input_geom.getAddressCovector()}, 
            output{ output_geom.getAddressCovector(), 
                    {output_geom.height, output_geom.width, output_geom.depth}},
            handler{ patch_config, agg_config, ot_config } {}
        
        Config(const FilterGeometry& filt,
               const T_acc * biases,
               const T_coef * kernel_tensor,
               const nn_acc32_to_int8_params_t* ot_params,
               const T_elm_in padding_value,
               const bool use_symmetric_saturation = false)
          : Config(filt.input, filt.output, T_patch::Config(filt.input, filt.window, padding_value), 
                   T_agg::Config(biases, kernel_tensor, filt.window),
                   T_ot::Config(ot_params, use_symmetric_saturation)) {}
    };




  protected:

    const Config config;

    struct {
      const T_elm_in* const input;
      T_elm_out* const output;
    } image;

  public:

    Conv2dDeepFilter(const Config& config,
                     const T_elm_in* input_image,
                     T_elm_out* output_image)
      : config(config), image{input_image, output_image} {}

    Conv2dDeepFilter(const T_elm_in* input_image,
                     T_elm_out* output_image,
                     const FilterGeometry& filt,
                     const T_acc * biases,
                     const T_coef * kernel_tensor,
                     const nn_acc32_to_int8_params_t* ot_params,
                     const bool use_symmetric_saturation = false)
      : config(filt, biases, kernel_tensor, ot_params, use_symmetric_saturation), 
        image{input_image, output_image} {}

    void execute(T_elm_in* patch_mem) const;

    JobClass spawnJob(
        const ImageRegion& region,
        T_elm_in* patch_mem) const;

};


}}}