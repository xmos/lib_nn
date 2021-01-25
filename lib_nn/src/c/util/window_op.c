
#include "window_op.h"

#include "../vpu_sim.h"

#include "../../asm/asm_constants.h"

#include <stdlib.h>
#include <string.h>




void perform_window_op(
    nn_image_t* p_Y,
    const nn_image_t* p_X,
    const nn_window_op_params_t* wop_params,
    const nn_window_op_job_params_t* job_params,
    nn_window_op_cb compute_output_callback,
    void* callback_context)
{

    nn_image_t (*Y)[wop_params->output.width][wop_params->output.channels] = (nn_image_t (*)[wop_params->output.width][wop_params->output.channels]) p_Y;
    nn_image_t (*X)[wop_params->input.width][wop_params->input.channels] = (nn_image_t (*)[wop_params->input.width][wop_params->input.channels]) p_X;

    const unsigned end_chan_out = job_params->start.channels + job_params->size.channels;
    const unsigned cout_tail = job_params->size.channels % VPU_INT8_ACC_PERIOD;
    const unsigned last_row = job_params->start.rows + job_params->size.rows - 1; 
    const unsigned last_col = job_params->start.cols + job_params->size.cols - 1; 


    // Iterate over channel output groups (COGs) first. This is sort of like computing the red channel outputs for all 
    // pixels before computing any of the green or blue channel outputs (except we're actually computing 16 output 
    // channels with each iteration, instead of just 1).
    // NOTE: I don't think this will work for maxpool2d because it computes 32 output channels at a time
    for(int cout = job_params->start.channels; cout < end_chan_out; cout += VPU_INT8_ACC_PERIOD){

        window_op_job_context_t job_context;
        job_context.flags = WIN_OP_FLAG_NONE;
        job_context.cur_out_chan = cout;

        // The number of output channels to be computed on this iteration
        job_context.out_chans = ((end_chan_out - cout) >= VPU_INT8_ACC_PERIOD)? VPU_INT8_ACC_PERIOD : cout_tail;

        // Iterate over the output rows for this COG
        for(int row = job_params->start.rows; row < job_params->start.rows + job_params->size.rows; row++){

            // Row (in X's coordinate space) of the first (top-left) cell of the convolution window
            job_context.window.row = wop_params->window.start.row + wop_params->window.stride.vertical * row;

            // Iterate over the output columns for this row
            for(int col = job_params->start.cols; col < job_params->start.cols + job_params->size.cols; col++){

                // Column (in X's coordinate space) of the first (top-left) cell of the convolution window
                job_context.window.col = wop_params->window.start.column + wop_params->window.stride.horizontal * col;

                // If this is the first pixel of the current COG, set the appropriate flag
                if((row == job_params->start.rows) && (col == job_params->start.cols))
                    job_context.flags |= WIN_OP_FLAG_FIRST_PIXEL_IN_COG;

                // If this is the final pixel of the current COG, set the appropriate flag
                if((row == last_row) && (col == last_col))
                    job_context.flags |= WIN_OP_FLAG_FINAL_PIXEL_IN_COG;

                int8_t output[VPU_INT8_EPV];

                // Compute the outputs
                compute_output_callback(&job_context, callback_context, output, 
                    &X[job_context.window.row][job_context.window.col][0], wop_params);

                // Copy results to output image
                for(int i = 0; i < job_context.out_chans; i++){
                    Y[row][col][cout + i] = output[i];
                }

            }
        }
    }
}

