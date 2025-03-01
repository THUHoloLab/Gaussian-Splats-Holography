// Authors: Shuhe Zhang, Yating Chen, Liangcai Cao
// Tsinghua University
// shuhe-zhang@tsinghua.edu.cn, clc@tsinghua.edu.cn
// forward propatation of rasterization

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
static __global__ void rasterize_forward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    // outputs
    int* __restrict__ final_index,
    float* __restrict__ out_img
);

static __global__ void rasterize_forward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    // outputs
    int* __restrict__ final_index,
    float* __restrict__ out_img
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;

    unsigned id_pix_x =
        block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned id_pix_y =
        block.group_index().y * block.group_dim().y + block.thread_index().y;       


    float px = (float) id_pix_x;
    float py = (float) id_pix_y;
    // int32_t pix_id = i * img_size.x + j;
    int32_t pix_id = id_pix_x * img_size.y + id_pix_y;
    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (id_pix_x < img_size.x) && (id_pix_y < img_size.y);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float pix_out = 0.f;
    // float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }
        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                        conic.y * delta.x * delta.y;
            const float alpha = min(1.f,  xy_opac.z * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            int32_t g = id_batch[t];
            const float c = colors[g];
            pix_out = pix_out + c * alpha;
            cur_idx = batch_start + t;
        }
        done = true;
    }
    if (inside) {
        final_index[pix_id] = cur_idx; // index of in bin of last gaussian in this pixel
        float final_color = pix_out; 
        out_img[pix_id] = final_color;
    }
}


void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *prhs[]
    // outputs params
    // mxGPUArray const *xys;          // prhs[0]
    // mxGPUArray const *conics;       // prhs[1]
    // mxGPUArray const *colors;       // prhs[2]
    // mxGPUArray const *opacities;    // prhs[3]
    // mxGPUArray const *gauss_ids;    // prhs[4]
    // mxGPUArray const *tile_bins;    // prhs[5]
    //                                 // img_sz prhs[6]    
) {
    // inputs
    mxGPUArray const *xys;          // prhs[0]
    mxGPUArray const *conics;       // prhs[1]
    mxGPUArray const *colors;       // prhs[2]
    mxGPUArray const *opacities;    // prhs[3]
    mxGPUArray const *gauss_ids;    // prhs[4]
    mxGPUArray const *tile_bins;    // prhs[5]
                                    // img_sz prhs[6]    
    float2 const *d_xys;
    float3 const *d_conics;
    float const *d_colors;
    float const *d_opacities;

    int32_t const *d_gauss_ids;
    int2 const *d_tile_bins;

    // outputs
    mxGPUArray *canvas; // output image
    mxGPUArray *final_idx;

    float *d_canvas;
    int *d_final_idx;
    /*************************************** */
    mxInitGPU();

    if (mxIsGPUArray(prhs[6])){
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input #4 must not be a GPU array");
    }

    xys       = mxGPUCreateFromMxArray(prhs[0]);
    conics    = mxGPUCreateFromMxArray(prhs[1]);
    colors    = mxGPUCreateFromMxArray(prhs[2]);
    opacities = mxGPUCreateFromMxArray(prhs[3]);
    gauss_ids = mxGPUCreateFromMxArray(prhs[4]);
    tile_bins = mxGPUCreateFromMxArray(prhs[5]);

    d_xys       = (float2 const *)(mxGPUGetDataReadOnly(xys));
    d_conics    = (float3 const *)(mxGPUGetDataReadOnly(conics));
    d_colors    = (float const *)(mxGPUGetDataReadOnly(colors));
    d_opacities = (float const *)(mxGPUGetDataReadOnly(opacities));
    d_gauss_ids = (int32_t const *)(mxGPUGetDataReadOnly(gauss_ids));
    d_tile_bins = (int2 const *)(mxGPUGetDataReadOnly(tile_bins));

    unsigned *img_size = (unsigned *) mxGetPr(prhs[6]); 
    unsigned img_h = img_size[0];
    unsigned img_w = img_size[1];

    mwSize arr_sz[] = {(size_t) img_h, (size_t) img_w};

    canvas = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(xys),arr_sz,
                                 mxGPUGetClassID(xys),mxGPUGetComplexity(xys),
                                 MX_GPU_INITIALIZE_VALUES);  

    final_idx = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(xys),arr_sz,
                                 mxINT32_CLASS,mxGPUGetComplexity(xys),
                                 MX_GPU_INITIALIZE_VALUES);  

    d_canvas = (float *)(mxGPUGetData(canvas));
    d_final_idx = (int *)(mxGPUGetData(final_idx));

    const dim3 BLOCK_DIM3 = {
        (unsigned) ((img_w + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((img_h + BLOCK_Y - 1) / BLOCK_Y),
        1
    };

    const dim3 THREAD_DIM3(BLOCK_X, BLOCK_Y, 1);
    const dim3 img_size_dim3(img_w, img_h, 1);

    rasterize_forward_kernel<<<BLOCK_DIM3, THREAD_DIM3>>>(
        BLOCK_DIM3,
        img_size_dim3,
        d_gauss_ids,
        d_tile_bins,

        d_xys,
        d_conics,
        d_colors,
        d_opacities,
        // outputs
        d_final_idx,
        d_canvas
    );
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(canvas);
    plhs[1] = mxGPUCreateMxArrayOnGPU(final_idx);

    mxGPUDestroyGPUArray(canvas);
    mxGPUDestroyGPUArray(final_idx);  

    mxGPUDestroyGPUArray(xys);
    mxGPUDestroyGPUArray(conics);
    mxGPUDestroyGPUArray(colors);
    mxGPUDestroyGPUArray(opacities);
    mxGPUDestroyGPUArray(gauss_ids);
    mxGPUDestroyGPUArray(tile_bins);
}