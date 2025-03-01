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

static __global__ void get_tile_bin_edges(
    const int num_intersects, 
    const int64_t* isect_ids_sorted, 
    int2* __restrict__ tile_bins
);

static __global__ void get_tile_bin_edges(
    const int num_intersects, 
    const int64_t* isect_ids_sorted, 
    int2* __restrict__ tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = (int32_t) 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile_idx].y = (int32_t) num_intersects;
    }
    if (idx == 0)
        return;
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = (int32_t) idx;
        tile_bins[cur_tile_idx].x = (int32_t) idx;
        return;
    }
}

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *prhs[]
    // num_intersects
    // isect_ids_sorted
){
    mxGPUArray const *isect_ids_sorted;
    int64_t const * d_isect_ids_sorted;

    mxGPUArray *tile_bins;
    int2 *d_tile_bins;

    mxInitGPU();

    if (!mxIsGPUArray(prhs[0])){
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input #2 must be a GPU array");
    }

    isect_ids_sorted   = mxGPUCreateFromMxArray(prhs[0]);
    d_isect_ids_sorted = (int64_t const *)(mxGPUGetDataReadOnly(isect_ids_sorted));

    size_t num_sects = mxGPUGetNumberOfElements(isect_ids_sorted);
    mwSize arr_sz[] = {(size_t) 2, num_sects};

    tile_bins = mxGPUCreateGPUArray(
                    mxGPUGetNumberOfDimensions(isect_ids_sorted),
                    arr_sz,
                    mxINT32_CLASS,
                    mxGPUGetComplexity(isect_ids_sorted),
                    MX_GPU_INITIALIZE_VALUES);  

    d_tile_bins = (int2 *) (mxGPUGetData(tile_bins));

    const unsigned N_BLOCKS = (unsigned) (((unsigned) num_sects + N_THREADS - 1) / N_THREADS);

    get_tile_bin_edges<<<N_BLOCKS, N_THREADS>>>(
        (int) num_sects,        // num_intersects,
        d_isect_ids_sorted,     // isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        d_tile_bins             // (int2 *)tile_bins.contiguous().data_ptr<int>()
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(tile_bins);

    mxGPUDestroyGPUArray(tile_bins);
    mxGPUDestroyGPUArray(isect_ids_sorted);
}