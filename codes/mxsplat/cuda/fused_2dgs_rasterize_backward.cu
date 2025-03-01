// Authors: Shuhe Zhang, Yating Chen, Liangcai Cao
// Tsinghua University
// shuhe-zhang@tsinghua.edu.cn, clc@tsinghua.edu.cn
// backward propatation of rasterization

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "config.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <algorithm>
#include <windowsnumerics.h>
#include <stdio.h>

namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}

static __global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const int* __restrict__ final_index,
    const float* __restrict__ v_canvas,         
    //const float* __restrict__ v_canvas_alpha,
    // outputs
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_colors,
    float* __restrict__ v_opacity
);

static __global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const int* __restrict__ final_index,
    const float* __restrict__ v_canvas,         
    //const float* __restrict__ v_canvas_alpha,
    // outputs
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_colors,
    float* __restrict__ v_opacity
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;

    unsigned id_pix_x =
        block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned id_pix_y =
        block.group_index().y * block.group_dim().y + block.thread_index().y;      

    const float px = (float)id_pix_x;
    const float py = (float)id_pix_y;
    // clamp this value to the last pixel
    const int32_t pix_id = id_pix_x * img_size.y + id_pix_y;//min(id_pix_x * img_size.y + id_pix_y, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (id_pix_x < img_size.x) && (id_pix_y < img_size.y);

    // this is the T AFTER the last gaussian in this pixel
    // float T_final = final_Ts[pix_id];
    // float T = T_final;
    // the contribution from gaussians behind the current one
    // float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float colors_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float v_out = v_canvas[pix_id];
    // const float v_out_alpha = v_canvas_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            colors_batch[tr] = colors[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                                      conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(1.f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float v_color_local = 0.f;
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // update v_rgb for this gaussian
                const float fac = alpha;
                float v_alpha = 0.f;
                v_color_local = fac * v_out;;

                const float this_color = colors_batch[t];
                // contribution from this pixel
                v_alpha += this_color * v_out;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                 1.0f * v_sigma * delta.x * delta.y, 
                                 0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                              v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_opacity_local = vis * v_alpha;
            }
            block.sync();
            warpSum(v_color_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            // block.sync();
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_colors_ptr = (float*)(v_colors);
                atomicAdd(v_colors_ptr + g + 0, v_color_local); // g will be the offset in RAM
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}



void mexFunction(
    /* @param output params */
    int nlhs, mxArray *plhs[],
    // mxGPUArray *dl_dxy; // output image
    // mxGPUArray *dl_dconic;
    // mxGPUArray *dl_dcolors;
    // mxGPUArray *dl_dopacity;

    /* @param input params */
    int nrhs, mxArray const *prhs[]
    // mxGPUArray const *gauss_ids;        // prhs[0]
    // mxGPUArray const *tile_bins;        // prhs[1]
    // mxGPUArray const *xys;              // prhs[2]    
    // mxGPUArray const *conics;           // prhs[3]
    // mxGPUArray const *colors;           // prhs[4]
    // mxGPUArray const *opacities;        // prhs[5]
    // mxGPUArray const *final_idx;        // prhs[6]
    // mxGPUArray const *v_canvas;         // prhs[7]
    // img_size;         // prhs[8]
) {
    // inputs
    mxGPUArray const *gauss_ids;        // prhs[0]
    mxGPUArray const *tile_bins;        // prhs[1]
    mxGPUArray const *xys;              // prhs[2]    
    mxGPUArray const *conics;           // prhs[3]
    mxGPUArray const *colors;           // prhs[4]
    mxGPUArray const *opacities;        // prhs[5]
    mxGPUArray const *final_idx;        // prhs[6]
    mxGPUArray const *v_canvas;         // prhs[7]
                                        // img_size prhs[8]
    float2 const *d_xys;
    float3 const *d_conics;
    float const *d_colors;
    float const *d_opacities;
    int32_t const *d_gauss_ids;
    int2 const *d_tile_bins;
    int const *d_final_idx;
    float const *d_v_canvas;

    // outputs
    mxGPUArray *dl_dxy; // output image
    mxGPUArray *dl_dconic;
    mxGPUArray *dl_dcolors;
    mxGPUArray *dl_dopacity;

    float2 *d_dl_dxy;
    float3 *d_dl_dconic;
    float *d_dl_dcolors, *d_dl_dopacity;
    /*************************************** */
    mxInitGPU();

    if (mxIsGPUArray(prhs[8])){
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input #9 must not be a GPU array");
    }

    gauss_ids       = mxGPUCreateFromMxArray(prhs[0]);
    tile_bins       = mxGPUCreateFromMxArray(prhs[1]);
    xys             = mxGPUCreateFromMxArray(prhs[2]);
    conics          = mxGPUCreateFromMxArray(prhs[3]);
    colors          = mxGPUCreateFromMxArray(prhs[4]);
    opacities       = mxGPUCreateFromMxArray(prhs[5]);
    final_idx       = mxGPUCreateFromMxArray(prhs[6]);
    v_canvas        = mxGPUCreateFromMxArray(prhs[7]);
    //v_canvas_alpha  = mxGPUCreateFromMxArray(prhs[8]);

    d_gauss_ids     = (int32_t const *)(mxGPUGetDataReadOnly(gauss_ids));
    d_tile_bins     = (int2 const *)(mxGPUGetDataReadOnly(tile_bins));
    d_xys           = (float2 const *)(mxGPUGetDataReadOnly(xys));
    d_conics        = (float3 const *)(mxGPUGetDataReadOnly(conics));
    d_colors        = (float const *)(mxGPUGetDataReadOnly(colors));
    d_opacities     = (float const *)(mxGPUGetDataReadOnly(opacities));
    d_final_idx     = (int const * )(mxGPUGetDataReadOnly(final_idx));
    d_v_canvas      = (float const * )(mxGPUGetDataReadOnly(v_canvas));
    //d_v_canvas_alpha = (float const * )(mxGPUGetDataReadOnly(v_canvas_alpha));

    unsigned const *img_size = (unsigned const *) mxGetPr(prhs[8]); 
    unsigned const img_h = img_size[0];
    unsigned const img_w = img_size[1];

    const dim3 BLOCK_DIM3 = {
        (unsigned) ((img_w + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((img_h + BLOCK_Y - 1) / BLOCK_Y),
        1
    };

    const dim3 THREAD_DIM3(BLOCK_X, BLOCK_Y, 1);
    const dim3 img_size_dim3(img_w, img_h, 1);

    /*********************** init output params **********************/
    dl_dxy = mxGPUCreateGPUArray(
                    mxGPUGetNumberOfDimensions(xys),
                    mxGPUGetDimensions(xys),
                    mxGPUGetClassID(xys),
                    mxGPUGetComplexity(xys),
                    MX_GPU_INITIALIZE_VALUES);  

    dl_dconic = mxGPUCreateGPUArray(
                    mxGPUGetNumberOfDimensions(conics),
                    mxGPUGetDimensions(conics),
                    mxGPUGetClassID(conics),
                    mxGPUGetComplexity(conics),
                    MX_GPU_INITIALIZE_VALUES);  

    dl_dcolors = mxGPUCreateGPUArray(
                    mxGPUGetNumberOfDimensions(colors),
                    mxGPUGetDimensions(colors),
                    mxGPUGetClassID(colors),
                    mxGPUGetComplexity(colors),
                    MX_GPU_INITIALIZE_VALUES);  
                                 
    dl_dopacity = mxGPUCreateGPUArray(
                    mxGPUGetNumberOfDimensions(opacities),
                    mxGPUGetDimensions(opacities),
                    mxGPUGetClassID(opacities),
                    mxGPUGetComplexity(opacities),
                    MX_GPU_INITIALIZE_VALUES);  

    d_dl_dxy        = (float2 *)(mxGPUGetData(dl_dxy));
    d_dl_dconic     = (float3 *)(mxGPUGetData(dl_dconic));
    d_dl_dcolors    = (float *)(mxGPUGetData(dl_dcolors));
    d_dl_dopacity   = (float *)(mxGPUGetData(dl_dopacity));
    /*********************** done! **********************/


    /*********************** launch the kernel !!! **********************/
    rasterize_backward_kernel<<<BLOCK_DIM3, THREAD_DIM3>>>(
        BLOCK_DIM3,      // const dim3 tile_bounds,
        img_size_dim3,   // const dim3 img_size,
        d_gauss_ids,     // const int32_t* __restrict__ gaussian_ids_sorted,
        d_tile_bins,     // const int2* __restrict__ tile_bins,
        d_xys,           // const float2* __restrict__ xys,
        d_conics,        // const float3* __restrict__ conics,
        d_colors,        // const float* __restrict__ colors,
        d_opacities,     // const float* __restrict__ opacities,
        d_final_idx,     // const int* __restrict__ final_index,
        d_v_canvas,      // const float* __restrict__ v_canvas,         
        /* output params */
        d_dl_dxy,// float2* __restrict__ v_xy,
        d_dl_dconic,// float3* __restrict__ v_conic,
        d_dl_dcolors, // float* __restrict__ v_colors,
        d_dl_dopacity// float* __restrict__ v_opacity   
    );
    /********** done! *********/

    plhs[0] = mxGPUCreateMxArrayOnGPU(dl_dxy);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dl_dconic);
    plhs[2] = mxGPUCreateMxArrayOnGPU(dl_dcolors);
    plhs[3] = mxGPUCreateMxArrayOnGPU(dl_dopacity);

    mxGPUDestroyGPUArray(dl_dxy);
    mxGPUDestroyGPUArray(dl_dconic);  
    mxGPUDestroyGPUArray(dl_dcolors);
    mxGPUDestroyGPUArray(dl_dopacity);  

    mxGPUDestroyGPUArray(gauss_ids);
    mxGPUDestroyGPUArray(tile_bins);
    mxGPUDestroyGPUArray(xys);
    mxGPUDestroyGPUArray(conics);
    mxGPUDestroyGPUArray(colors);
    mxGPUDestroyGPUArray(opacities);
    mxGPUDestroyGPUArray(final_idx);
    mxGPUDestroyGPUArray(v_canvas);
}