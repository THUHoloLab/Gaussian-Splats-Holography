// Authors: Shuhe Zhang, Yating Chen, Liangcai Cao
// Tsinghua University
// shuhe-zhang@tsinghua.edu.cn, clc@tsinghua.edu.cn
//

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <algorithm>
#include <windowsnumerics.h>

namespace cg = cooperative_groups;

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2 &bb_min,
    uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 g_center,
    const float g_radius,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        g_center.x / (float)BLOCK_X, 
        g_center.y / (float)BLOCK_Y
    };

    float2 tile_radius = {
        g_radius / (float)BLOCK_X, 
        g_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

static __global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);
// kernel function for projecting each gaussian on device
// each thread processes one gaussian
// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
static __global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian
    uint2 tile_min, tile_max;
    float2 center = xys[idx];
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max);
    // tile_min.x, tile_min.y, tile_max.x, tile_max.y);

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : (cum_tiles_hit[idx - 1]);

    int64_t depth_id = (int64_t) * (int32_t *)&(depths[idx]);
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as int32
            int64_t tile_id = i * tile_bounds.x + j; // tile within image
            isect_ids[cur_idx] = ((tile_id << 32) | depth_id); // tile | depth id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
    // printf("point %d ending at %d\n", idx, cur_idx);
}

void mexFunction(
    int nlhs, mxArray *plhs[],
    // gaussian_ids_sorted,
    // tile_bins,
    int nrhs, mxArray const *prhs[]
    // num_intersects,
    // xys, 2 * N array
    // depths,
    // radii,
    // cum_tiles_hit, 
    // img_size, 
) {
    // inputs
    mxGPUArray const *xys, *depths, *radii, *cum_tiles_hit;
    float2 const *d_xys;
    float const *d_depths;
    int const *d_radii;
    int32_t const *d_cum_tiles_hit; // 2 X N params, N = number of Gaussian

    // outputs 
    mxGPUArray *isect_ids_unsorted, *gaussian_ids_unsorted;
    int64_t *d_isect_ids_unsorted;
    int32_t *d_gaussian_ids_unsorted;

    mxInitGPU();

    if (mxIsGPUArray(prhs[0])){
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input #1 must not be a GPU array");
    }

    if (mxIsGPUArray(prhs[5])){
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input #6 must not be a GPU array");
    }
    
    unsigned const *img_size = (unsigned const*) mxGetPr(prhs[5]); 
    unsigned const img_h = img_size[0];
    unsigned const img_w = img_size[1];

    const dim3 tile_bounds = {
        (unsigned) ((img_w + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((img_h + BLOCK_Y - 1) / BLOCK_Y),
        1
    };

    xys             = mxGPUCreateFromMxArray(prhs[1]);
    depths          = mxGPUCreateFromMxArray(prhs[2]);
    radii           = mxGPUCreateFromMxArray(prhs[3]);
    cum_tiles_hit   = mxGPUCreateFromMxArray(prhs[4]);

    d_xys               = (float2 const *)(mxGPUGetDataReadOnly(xys));
    d_depths            = (float const *)(mxGPUGetDataReadOnly(depths));
    d_radii             = (int const *)(mxGPUGetDataReadOnly(radii));
    d_cum_tiles_hit     = (int32_t const *)(mxGPUGetDataReadOnly(cum_tiles_hit));

    size_t num_points = mxGPUGetNumberOfElements(depths);
    const unsigned N_BLOCKS = (unsigned) ((unsigned) num_points + N_THREADS - 1) / N_THREADS;

    int32_t *num_intersects = (int32_t *) mxGetPr(prhs[0]);
    size_t num_sects = num_intersects[0];
    // printf("num_sects: %d",num_sects);

    mwSize arr_sz[] = {(size_t) 1, num_sects};
    isect_ids_unsorted = mxGPUCreateGPUArray(
                            mxGPUGetNumberOfDimensions(depths),
                            arr_sz,
                            mxINT64_CLASS,
                            mxGPUGetComplexity(depths),
                            MX_GPU_INITIALIZE_VALUES);  

    gaussian_ids_unsorted = mxGPUCreateGPUArray(
                            mxGPUGetNumberOfDimensions(depths),
                            arr_sz,
                            mxINT32_CLASS,
                            mxGPUGetComplexity(depths),
                            MX_GPU_INITIALIZE_VALUES);  

    d_isect_ids_unsorted       = (int64_t *)(mxGPUGetData(isect_ids_unsorted));
    d_gaussian_ids_unsorted    = (int32_t *)(mxGPUGetData(gaussian_ids_unsorted));

    map_gaussian_to_intersects<<<N_BLOCKS, N_THREADS>>>(
        (int) num_points,   
        d_xys,  
        d_depths, 
        d_radii,   
        d_cum_tiles_hit, 
        tile_bounds, 
        // // Outputs.
        d_isect_ids_unsorted, 
        d_gaussian_ids_unsorted
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(isect_ids_unsorted);
    plhs[1] = mxGPUCreateMxArrayOnGPU(gaussian_ids_unsorted);

    mxGPUDestroyGPUArray(isect_ids_unsorted);
    mxGPUDestroyGPUArray(gaussian_ids_unsorted);

    mxGPUDestroyGPUArray(xys);
    mxGPUDestroyGPUArray(depths);
    mxGPUDestroyGPUArray(radii);
    mxGPUDestroyGPUArray(cum_tiles_hit);
}
