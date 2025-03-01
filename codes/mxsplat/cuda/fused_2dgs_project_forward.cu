// Authors: Shuhe Zhang, Yating Chen, Liangcai Cao
// Tsinghua University
// shuhe-zhang@tsinghua.edu.cn, clc@tsinghua.edu.cn
//
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "config.h"

#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <algorithm>
#include <windowsnumerics.h>
#include <stdio.h>

namespace cg = cooperative_groups;

inline __device__ bool compute_cov2d_bounds(
    const float3 cov2d, 
    float3 &conic, 
    float &radius
) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x =  cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z =  cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

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

inline __device__ glm::mat2 rotmat2d(const float rot) {
    // quat to rotation matrix
    float cosr = cosf(rot);
    float sinr = sinf(rot);

    glm::mat2 R = glm::mat2(cosr);
    R[0][1] = -sinr;
    R[1][0] =  sinr;

    // glm matrices are column-major
    return R;
}

inline __device__ glm::mat2 scale_to_mat2d(const float2 scale) {
    glm::mat2 S = glm::mat2(1.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    return S;
}
static __global__ void project_2d_Gaussians_forward_kernel(
    const int num_points,
    const float2* __restrict__ xy_pos,
    const float* __restrict__ rot,
    const float2* __restrict__ scales,
    const int img_w,
    const int img_h,
    const dim3 tile_bounds,
    // output params
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
);

static __global__ void project_2d_Gaussians_forward_kernel(
    const int num_points,
    const float2* __restrict__ xy_pos,
    const float* __restrict__ rot,
    const float2* __restrict__ scales,
    const int img_w,
    const int img_h,
    const dim3 tile_bounds,
    // output params
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
    // Retrieve the 2D Gaussian parameters
    // 直接以图像像素做归一化，x 和 y 的范围是 [-1, 1].
    float2 center = {0.5f * img_w * xy_pos[idx].x + 0.5f * img_w,
                     0.5f * img_h * xy_pos[idx].y + 0.5f * img_h};

    glm::mat2 R = rotmat2d(rot[idx]);
    glm::mat2 S = scale_to_mat2d(scales[idx]);
    glm::mat2 M = R * S;
    glm::mat2 tmp = M * glm::transpose(M);
        // glm::mat2 tmp = R * S * glm::transpose(R);
    float3 cov2d = make_float3(tmp[0][0], tmp[0][1], tmp[1][1]);
        // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float radius; // the radius that a single Gaussian covers and valished at 6 sigma
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return;
    conics[idx] = conic;

    xys[idx] = center;

    radii[idx] = (int) radius;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        return;
    }
    num_tiles_hit[idx] = tile_area;
    depths[idx] = 0.0f;
}

/**
 * @brief .Initialize the Kernel function for forward projection of Gaussians
 *
 *
 * We recommend that you call mxInitGPU at the beginning of your MEX file unless
 * you have an alternate way of guaranteeing that the MathWorks GPU API has
 * been initialized at the start of your MEX file.
 *
 * If the library is already initialized, this function returns without doing
 * any work. If the library has not been initialized, the function initializes
 * the default device.
 * @note At present, MATLAB can only work with one GPU device at a time.

 * @param input_params  mxArray *plhs[] parameters on the left-hand side;
        xy_pos,     //float2* __restrict__ xy_pos,
        rot,        //float* __restrict__ rot,
        scales,     //float2* __restrict__ scales,
        img_size,   //float* __restrict__ img_size,
 * @returns mxArray *prhs[] parameters on the right-hand side;
        xys,          //float2* __restrict__ xys,
        depths,       //float* __restrict__ depths,
        radii,        //int* __restrict__ radii,
        conics,       //float3* __restrict__ conics,
        num_tiles_hit //float* __restrict__ num_tiles_hit
 */

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, mxArray const *prhs[]
) {
    // input params 4
    mxGPUArray const *xy_pos, *rot, *scales;
    float2 const *d_xy_pos,*d_scales;  // 2 X N params, N = number of Gaussian
    float const *d_rot;

    // prhs[3] is the image size [h,w]

    // output params 5
    mxGPUArray *xys, *depths, *radii, *conics, *num_tiles_hit;
    float2 *d_xys;
    float3 *d_conics;  // 3 X N conics params, N = number of Gaussian
    float *d_depths;
    int *d_radii;
    int32_t *d_num_tiles_hit;

    mxInitGPU();

    if (nrhs!=4) {
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input must have 4 inputs.");
    }

    if ((!mxIsGPUArray(prhs[0])) || (!mxIsGPUArray(prhs[1])) || (!mxIsGPUArray(prhs[2]))) {
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The first 3 inputs must all be GPU arrays.");
    }

    if (mxIsGPUArray(prhs[3])){
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input #4 must not be a GPU array");
    }

    xy_pos  = mxGPUCreateFromMxArray(prhs[0]);
    rot     = mxGPUCreateFromMxArray(prhs[1]);
    scales  = mxGPUCreateFromMxArray(prhs[2]);

    d_xy_pos  = (float2 const *)(mxGPUGetDataReadOnly(xy_pos));
    d_scales  = (float2 const *)(mxGPUGetDataReadOnly(scales));
    d_rot     = (float const *)(mxGPUGetDataReadOnly(rot));

    unsigned *img_size = (unsigned *) mxGetPr(prhs[3]); 
    unsigned img_h = img_size[0];
    unsigned img_w = img_size[1];
    
    // init output parameters
    xys           = mxGPUCopyFromMxArray(prhs[0]);
    depths        = mxGPUCopyFromMxArray(prhs[1]);
    //radii         = mxGPUCopyFromMxArray(prhs[1]);
 
    size_t num_points = mxGPUGetNumberOfElements(rot); // Number of Gaussian points

    mwSize arr_sz[] = {(size_t) 3, num_points};
    radii = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(rot),mxGPUGetDimensions(rot),
                                 mxINT32_CLASS,mxGPUGetComplexity(rot),
                                 MX_GPU_INITIALIZE_VALUES);  

    conics = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(xy_pos),arr_sz,
                                 mxGPUGetClassID(xy_pos),mxGPUGetComplexity(xy_pos),
                                 MX_GPU_INITIALIZE_VALUES);  

    num_tiles_hit = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(rot),mxGPUGetDimensions(rot),
                                 mxINT32_CLASS,mxGPUGetComplexity(rot),
                                 MX_GPU_INITIALIZE_VALUES);  

    d_xys           = (float2 *)(mxGPUGetData(xys));
    d_depths        = (float *)(mxGPUGetData(depths));
    d_radii         = (int *)(mxGPUGetData(radii));
    d_conics        = (float3 *)(mxGPUGetData(conics));
    d_num_tiles_hit = (int32_t *)(mxGPUGetData(num_tiles_hit));

    const unsigned N_BLOCKS = (unsigned) ((unsigned) num_points + N_THREADS - 1) / N_THREADS;

    const dim3 tile_bounds = {
        (unsigned) ((img_w + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((img_h + BLOCK_Y - 1) / BLOCK_Y),
        1
    };

    // printf("tile bounds x: %d, y: %d", (int) tile_bounds.x, (int) tile_bounds.y);
    project_2d_Gaussians_forward_kernel<<<N_BLOCKS, N_THREADS>>>(
        (int) num_points,
        d_xy_pos,           
        d_rot,           
        d_scales,        
        (int) img_w,        
        (int) img_h,       
        tile_bounds,      
        // output params
        d_xys,       
        d_depths,     
        d_radii,       
        d_conics,      
        d_num_tiles_hit
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(xys);
    plhs[1] = mxGPUCreateMxArrayOnGPU(depths);
    plhs[2] = mxGPUCreateMxArrayOnGPU(radii);
    plhs[3] = mxGPUCreateMxArrayOnGPU(conics);
    plhs[4] = mxGPUCreateMxArrayOnGPU(num_tiles_hit);

    mxGPUDestroyGPUArray(xy_pos);
    mxGPUDestroyGPUArray(rot);
    mxGPUDestroyGPUArray(scales);

    mxGPUDestroyGPUArray(xys);
    mxGPUDestroyGPUArray(depths);
    mxGPUDestroyGPUArray(radii);
    mxGPUDestroyGPUArray(conics);
    mxGPUDestroyGPUArray(num_tiles_hit);
}