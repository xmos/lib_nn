
#include "window_op.h"

#include "../vpu_sim.h"

#include "../../asm/asm_constants.h"

#include <stdlib.h>
#include <string.h>



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
    const nn_window_op_params_t* wop_params)
{

    conv2d_patch_handler_context_t* context = (conv2d_patch_handler_context_t*) p_context;

    // The accumulator will accumulate into here.
    nn_acc32_vector_t __attribute__((aligned (4))) acc;

    // Accumulate over the patch
    context->accumulator(X, context->acc_context, &acc, job_context, wop_params);

    // Resolve the 32-bit accumulators to get the final 8-bit outputs
    context->acc_resolver(Y, &acc, context->res_context, job_context, wop_params);

}


/**
 * Accumulator callback for dense (as opposed to e.g. depthwise) inner products of a patch with a convolution kernel
 */
void conv2d_patch_accumulator_dense_ref(
    const nn_image_t* X,
    const nn_conv2d_accumulator_params_t* acc_context,
    nn_acc32_vector_t* acc,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{
    xs3_vpu vpu;

    // Set VPU to 8-bit mode
    VSETC(&vpu, MODE_S8);
    
    // The cost of doing the math to move the K and bias pointers each time this callback is called is probably small
    // compared to the overall cost of doing the accumulation, so the simplicity of not having to coordinate increments
    // to those pointers after each COG is processed is probably worth it.
    
    // The COG we're currently processing. The bias and K blocks need to be advanced based on this.
    const int cur_cog = job_context->cur_out_chan >> VPU_INT8_ACC_PERIOD_LOG2;

    // It will probably save time and maybe even memory to just cache this in the acc_context
    // (Also the size of the convolution window in bytes)
    const mem_stride_t cout_slice_bytes = wop_params->window.shape.height 
                                        * wop_params->window.shape.width
                                        * wop_params->input.channels;
                                        
    // Bytes to increment K to advance an entire COG
    const mem_stride_t cog_slice_bytes = cout_slice_bytes << VPU_INT8_ACC_PERIOD_LOG2;


    // Initialize the accumulators
    if(1) {
        const nn_acc32_vector_t* bias = acc_context->bias;

        bias = &bias[cur_cog];

        VLDD(&vpu, bias->high);
        VLDR(&vpu, bias->low);

        VSTD(&vpu, acc->high);
        VSTR(&vpu, acc->low);
    }

    // Get a reference to the current COG slice from K
    const int8_t* K = &acc_context->K[cur_cog * cog_slice_bytes];


    // Offset for loading accumulators into vD:vR. (note, this is the number of int16_ts to offset by, it should
    // be doubled for the number of bytes
    const int acc_ld_offset = VPU_INT8_ACC_PERIOD - job_context->out_chans;



    // The number of bytes in one cout slice of the kernel tensor is the same as the size of the convolution window
    //  (K_height * K_width * X_channels)
    const int conv_window_bytes = wop_params->input.channels 
                                * wop_params->window.shape.height 
                                * wop_params->window.shape.width;

    // Round up. The scratch buffer should be padded out at the end with zeros                                          
    const unsigned macc_groups = (cout_slice_bytes + (VPU_INT8_EPV - 1)) >> VPU_INT8_EPV_LOG2;


    // Because we have to iterate through the output channels in reverse order (but the channel input groups
    // in forward order) we need to move K to point at the *start* of the final channel of the COG.
    K = &K[(job_context->out_chans-1) * cout_slice_bytes];

    for(int i = 0; i < macc_groups; i++){

        //First, load the accumulators into vD:vR with the alignment dictated by the number of output channels
        // currently being processed.
        VLDD(&vpu, &acc->high[-acc_ld_offset]);
        VLDR(&vpu, &acc->low[-acc_ld_offset]); 

        VLDC(&vpu, &X);

        for(int cout = job_context->out_chans; cout >= 0; cout--){

            // Accumulate
            VLMACCR(&vpu, K);

            // Move K back one output channel
            K = &K[-conv_window_bytes];
        }

        VSTD(&vpu, &acc->high[0]);
        VSTD(&vpu, &acc->low[0]);

        // We've gone through the above loop out_chans times, so we've subtracted out_chans * conv_window_bytes from
        // K. Add that back plus VPU_INT8_EPV to move to the next CIG.
        K = &K[(job_context->out_chans * conv_window_bytes) + VPU_INT8_EPV];

        // Move to the next CIG of the input.
        X = &X[VPU_INT8_EPV];

    }
}



void conv2d_acc32_symmetric_resolver_ref(
    nn_image_t* Y,
    const nn_acc32_vector_t* acc,
    const nn_acc32_to_int8_params_t* op_params,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{
    xs3_vpu vpu;
    vpu_vector_t vec_tmp;

    // Move op_params to point at the current COG's params
    op_params = &op_params[job_context->cur_out_chan >> VPU_INT8_ACC_PERIOD];


    VSETC(&vpu, MODE_S16);

    VLDD(&vpu, acc->high);
    VLDR(&vpu, acc->low);

    VLSAT(&vpu, op_params->shift1);

    VSTR(&vpu, &vec_tmp);
    VLDC(&vpu, op_params->scale);
    VCLRDR(&vpu);
    VLMACC(&vpu, &vec_tmp);
    VLDC(&vpu, op_params->offset_scale);
    VLMACC(&vpu, op_params->offset);

    VSETC(&vpu, MODE_S8);

    VLSAT(&vpu, op_params->shift2);

    VSTR(&vpu, Y);
}



void conv2d_acc32_asymmetric_resolver_ref(
    nn_image_t* Y,
    const nn_acc32_vector_t* acc,
    const nn_acc32_to_int8_params_t* op_params,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{
    xs3_vpu vpu;
    vpu_vector_t vec_tmp;

    // Move op_params to point at the current COG's params
    op_params = &op_params[job_context->cur_out_chan >> VPU_INT8_ACC_PERIOD];

    VLDR(&vpu, vpu_vects.vec_0x80);
    VSTR(&vpu, Y);

    VSETC(&vpu, MODE_S16);

    VLDD(&vpu, acc->high);
    VLDR(&vpu, acc->low);

    VLSAT(&vpu, op_params->shift1);

    VSTR(&vpu, &vec_tmp);
    VLDC(&vpu, op_params->scale);
    VCLRDR(&vpu);
    VLMACC(&vpu, &vec_tmp);
    VLDC(&vpu, op_params->offset_scale);
    VLMACC(&vpu, op_params->offset);

    VLSAT(&vpu, op_params->shift2);

    VSTR(&vpu, &vec_tmp);
    VLADD(&vpu, vpu_vects.vec_0x007F);
    VDEPTH1(&vpu);
    uint32_t mask = ~vpu.vR.s32[0];

    VLASHR(&vpu, &vec_tmp, -8);
    VDEPTH8(&vpu);
    VSTRPV(&vpu, Y, mask);
}



#ifdef NN_USE_REF

void conv2d_patch_accumulator_dense(
    const nn_image_t* X,
    const nn_conv2d_accumulator_params_t* acc_context,
    nn_acc32_vector_t* acc,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{
    conv2d_patch_accumulator_dense_ref(p_X, p_k, acc_hi, acc_lo, job_context, wop_params);
}

void conv2d_acc32_symmetric_resolver(
    nn_image_t* Y,
    const nn_acc32_vector_t* acc,
    const nn_acc32_to_int8_params_t* op_params,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{
    conv2d_acc32_symmetric_resolver_ref(BSO, Y, acc_hi, acc_lo);
}

void conv2d_acc32_symmetric_resolver(
    nn_image_t* Y,
    const nn_acc32_vector_t* acc,
    const nn_acc32_to_int8_params_t* op_params,
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{
    conv2d_acc32_symmetric_resolver_ref(BSO, Y, acc_hi, acc_lo);
}

#endif // NN_USE_REF