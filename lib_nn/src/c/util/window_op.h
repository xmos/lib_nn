
#ifndef WINDOW_OP_H_
#define WINDOW_OP_H_

#include "nn_operator.h"


typedef enum {
    WIN_OP_FLAG_NONE = 0,
    WIN_OP_FLAG_FIRST_PIXEL_IN_COG = 1,
    WIN_OP_FLAG_FINAL_PIXEL_IN_COG = 2,
} window_op_flags_e;



// Pretty much every function and callback had each of these as a parameter, so I just combined
// them into one struct that gets passed around
typedef struct {
    nn_image_params_t input;
    nn_image_params_t output;
    nn_window_params_t window;
} nn_window_op_params_t;



typedef struct {
    uint16_t shift1[ VPU_INT8_ACC_PERIOD ];
    int16_t scale[ VPU_INT8_ACC_PERIOD ];
    int16_t offset_scale[ VPU_INT8_ACC_PERIOD ];
    int16_t offset[ VPU_INT8_ACC_PERIOD ];
    uint16_t shift2[ VPU_INT8_ACC_PERIOD ];
} nn_acc32_to_int8_params_t;



// To determine things like whether a given cell of a patch is padding, the top-level function (perform_window_op())
// needs to provide a way for the callback to determine where the window is currently located. This is how it does 
// that.
typedef struct {
    // The location of the first cell of the convolution window, in X's coordinate space. If these are negative, it 
    // means there is top or left padding.
    struct {
        int row;
        int col;
    } window;
    unsigned cur_out_chan;
    // Number of output channels currently being computed
    channel_count_t out_chans;
    // Flags signaled from doWindowOp() to the callback
    window_op_flags_e flags;
} window_op_job_context_t;







/* e.g.
    void my_window_op_callback(
        window_op_job_context_t* job_context, 
        void* callback_context,
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_window_op_params_t* wop_params)
*/
typedef void (*nn_window_op_cb)(
    const window_op_job_context_t* job_context,
    void* handler_context,
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_window_op_params_t* wop_params);




void perform_window_op(
    nn_image_t* p_Y,
    const nn_image_t* p_X,
    const nn_window_op_params_t* wop_params,
    const nn_window_op_job_params_t* job_params,
    nn_window_op_cb compute_output_callback,
    void* callback_context);




/*
 * im2col_window_op() takes a callback function (called its patch handler) as part of its context. 
 * This is the required signature.
 * 
 * The callback is given an output pointer Y to be populated, a pointer to a contiguous block of memory which is 
 * essentially the intersection of the (convolution or pooling) window and the input image X. If the convolution window
 * extends outside the bounds of the input image X, padding will be used for those regions of the patch.
 * 
 * Because the callback will need instance-specific parameters (e.g. convolution kernel tensor), im2col_context_t also
 * contains an opaque pointer (im2col_context.patch_handler_context) which is passed along to the callback.
 * 
 * The callback is also passed the nn_window_op_params_t pointer that was given to im2col_window_op() which contains
 * info about the structure of the input image, output image and operation window.
 */
typedef void (*im2col_patch_handler_cb)( 
    void* context,
    nn_image_t* Y,
    const nn_image_t* patch,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params);



/*
 * Constructed by the user of perform_window_op()
 */
typedef struct {
    int8_t padding_value;   // Value used in the patch buffer where the window extends beyond the bounds of X
    nn_image_t* scratch;    // patch buffer. Must be at least the size of a patch (window height * width * x channels)
                            // Although some patch handlers (e.g. conv2d) might require it to be slightly larger
    void* patch_handler_context; // Opaque context passed to the patch handler
    im2col_patch_handler_cb patch_handler; // patch handler callback
    
} im2col_context_t;


/**
 * 
 */
void im2col_window_op(
    const window_op_job_context_t* job_context,
    void* p_im2col_context,
    nn_image_t Y[VPU_INT8_ACC_PERIOD],
    const nn_image_t* p_X,
    const nn_window_op_params_t* wop_params);





/**
 * conv2d_patch_handler() takes two callbacks. The first is a delegate called to actually perform the accumulation
 * into the accumulators. The second is a delegate which takes the 32-bit accumulators and resolves the 8-bit results
 * from them.
 * 
 */
void conv2d_patch_handler(
    void* p_context,  
    nn_image_t* Y, 
    const nn_image_t* X, 
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params);



typedef struct {
    int16_t high[ VPU_INT8_ACC_PERIOD ];
    uint16_t low[ VPU_INT8_ACC_PERIOD ];
} nn_acc32_vector_t;


typedef struct {
    nn_acc32_vector_t* bias;
    nn_tensor_t* K;
} nn_conv2d_accumulator_params_t;

/**
 * Function signature for the conv2d_patch_handler() accumulator callback.
 */
typedef void (*conv2d_patch_accumulator_cb)(
    const nn_image_t* patch,
    const nn_conv2d_accumulator_params_t* acc_context,
    nn_acc32_vector_t* accumulators,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_param); 




/**
 * Function signature for callback which resolves 32-bit accumulators to 8-bit final
 */
typedef void (*acc_resolver_cb)(
    nn_image_t* Y,
    const nn_acc32_vector_t* accumulators,
    const nn_acc32_to_int8_params_t* op_params,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params);





/**
 * Context struct for conv2d_patch_handler()
 */
typedef struct {
    conv2d_patch_accumulator_cb accumulator;
    acc_resolver_cb acc_resolver;
    nn_conv2d_accumulator_params_t* acc_context;
    nn_acc32_to_int8_params_t* res_context;
} conv2d_patch_handler_context_t;



/**
 * Accumulator callback for dense (as opposed to depthwise) inner products of a patch with a convolution kernel
 */
void conv2d_patch_accumulator_dense(
    const nn_image_t* p_X,
    const nn_conv2d_accumulator_params_t* acc_context,
    nn_acc32_vector_t* accumulators,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params);


/**
 * 
 */
void conv2d_acc32_symmetric_resolver(
    nn_image_t* Y,
    const nn_acc32_vector_t* accumulators,
    const nn_acc32_to_int8_params_t* op_params,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params);


/**
 * 
 */
void conv2d_acc32_asymmetric_resolver(
    nn_image_t* Y,
    const nn_acc32_vector_t* accumulators,
    const nn_acc32_to_int8_params_t* op_params,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params);


/**
 * 
 */
void maxpool2d_patch_handler(
    void* p_context,  
    nn_image_t* Y, 
    const nn_image_t* X, 
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params);


#endif // WINDOW_OP_H_