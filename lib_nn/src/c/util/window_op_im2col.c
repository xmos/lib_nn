
#include "window_op.h"

#include "../vpu_sim.h"

#include "../../asm/asm_constants.h"

#include <stdlib.h>
#include <string.h>




void im2col_window_op(
    const window_op_job_context_t* job_context,
    void* p_im2col_context,
    nn_image_t Y[VPU_INT8_ACC_PERIOD],
    const nn_image_t* p_X,
    const nn_window_op_params_t* wop_params)
{    
    im2col_context_t* im2col_context = (im2col_context_t*) p_im2col_context;

    // Cast the X pointer to be the same (3D) shape as X.
    // (Note: X points at the start of the convolution window, not the start of the image)
    nn_image_t (*X)[wop_params->input.width][wop_params->input.channels] = 
        (nn_image_t (*)[wop_params->input.width][wop_params->input.channels]) p_X;

    // Cast the scratch pointer to the (3D) shape of the convolution window
    nn_image_t (*scratch)[wop_params->window.shape.width][wop_params->input.channels] = 
        (nn_image_t (*)[wop_params->window.shape.width][wop_params->input.channels]) im2col_context->scratch;

    // Fill the scratch buffer before dealing with the accumulators
    for(int row = 0; row < wop_params->window.shape.height; row++){

        // Current row of convolution window (in X's coordinate space)
        const int x_row = job_context->window.row + row * wop_params->window.dilation.vertical;

        // Check if this whole row of the convolution window is in padding
        if( (x_row < 0) || (x_row >= wop_params->input.height) ){
            // Row is in padding. Set the corresponding region of the scratch buffer to the zero point, and then
            // continue to next row.
            memset(&scratch[row][0][0], im2col_context->padding_value, 
                    wop_params->input.channels * wop_params->window.shape.width);
            continue;
        }

        for(int col = 0; col < wop_params->window.shape.width; col++){

            // Current column of convolution window (in X's coordinate space) 
            const int x_col = job_context->window.col + col * wop_params->window.dilation.horizontal;

            if( (x_col < 0) || (x_col >= wop_params->input.width) ){
                // Column is in padding. Set the corresponding region of the scratch buffer to the zero point, and
                // then continue to the next column.
                memset(&scratch[row][col][0], im2col_context->padding_value, wop_params->input.channels);
                continue;
            }

            // Copy pixel from X into the scratch buffer.
            memcpy(&scratch[row][col][0], &X[row][col][0], wop_params->input.channels);
        }
    }

    //Call the patch handler
    im2col_context->patch_handler(im2col_context->patch_handler_context, 
        Y, im2col_context->scratch, job_context, wop_params);

}

    