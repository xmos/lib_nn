
#include "window_op.h"

#include "../vpu_sim.h"

#include "../../asm/asm_constants.h"

#include <stdlib.h>
#include <string.h>



void maxpool2d_patch_handler_ref(
    void* p_context,  
    nn_image_t* Y, 
    const nn_image_t* X, 
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{

    xs3_vpu vpu;
    vpu_vector_t* vec_curmax = (vpu_vector_t*) Y;
    vpu_vector_t vec_tmp;

    VSETC(&vpu, MODE_S8);

    // Preload outputs with -128
    VLDR(&vpu, vpu_vects.vec_0x80);
    VSTR(&vpu, vec_curmax);

    const unsigned patch_pixels = wop_params->window.shape.height * wop_params->window.shape.width;

    for(int i = 0; i < patch_pixels; i++){

        VLDR(&vpu, X);
        VLSUB(&vpu, vec_curmax);
        VDEPTH1(&vpu);
        uint32_t mask = vpu.vR.s32[0];

        VLDR(&vpu, X);
        VSTRPV(&vpu, vec_curmax, mask);

        X = &X[ wop_params->input.channels ];

    }
}



#ifdef NN_USE_REF

void maxpool2d_patch_handler(
    void* p_context,  
    nn_image_t* Y, 
    const nn_image_t* X, 
    const window_op_job_context_t* job_context,
    const nn_window_op_params_t* wop_params)
{
    maxpool2d_patch_handler_ref(p_context, Y, X, job_context, wop_params);
}


#endif // NN_USE_REF