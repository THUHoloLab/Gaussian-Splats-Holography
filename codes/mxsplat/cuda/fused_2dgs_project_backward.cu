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
    // take 6 sigma of covariance
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

inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

inline __device__ glm::mat2 rotmat2d_gradient(const float rot) {
    // quat to rotation matrix
    float cosr = cos(rot);
    float sinr = sin(rot);

    glm::mat2 R = glm::mat2(-sinr);
    R[0][1] = -cosr;
    R[1][0] = cosr;

    // glm matrices are column-major
    return R;
}
static __global__ void project_2d_Gaussians_backward_kernel(
    const int num_points,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ dv_cov2d,
    float2* __restrict__ dv_xy_pos,
    float2* __restrict__ dv_scale,
    float* __restrict__ dv_rot
);
static __global__ void project_2d_Gaussians_backward_kernel(
    const int num_points,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ dv_cov2d,
    float2* __restrict__ dv_xy_pos,
    float2* __restrict__ dv_scale,
    float* __restrict__ dv_rot
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], dv_cov2d[idx]);

    glm::mat2 R = rotmat2d(rotation[idx]);
    glm::mat2 R_g = rotmat2d_gradient(rotation[idx]);
    glm::mat2 S = scale_to_mat2d(scales2d[idx]);
    glm::mat2 M = R * S;
    glm::mat2 theta_g = R_g * S * glm::transpose(M) + M * glm::transpose(S) * glm::transpose(R_g);
    
    glm::mat2 scale_x_g = glm::mat2(0.f);
    scale_x_g[0][0] = 2.f * scales2d[idx].x;
    glm::mat2 scale_y_g = glm::mat2(0.f);
    scale_y_g[1][1] = 2.f * scales2d[idx].y;

    glm::mat2 sigma_x_g = R * scale_x_g * glm::transpose(R);
    glm::mat2 sigma_y_g = R * scale_y_g * glm::transpose(R);

    float G_11 = dv_cov2d[idx].x; // dL/dSigma_11
    float G_12 = dv_cov2d[idx].y; // dL/dSigma_12, which is the same as dL/dSigma_21
    float G_22 = dv_cov2d[idx].z; // dL/dSigma_22

    dv_scale[idx].x = G_11 * sigma_x_g[0][0] + 2 * G_12 * sigma_x_g[0][1] + G_22 * sigma_x_g[1][1];
    dv_scale[idx].y = G_11 * sigma_y_g[0][0] + 2 * G_12 * sigma_y_g[0][1] + G_22 * sigma_y_g[1][1];
    dv_rot[idx]     = G_11 * theta_g[0][0] + 2 * G_12 * theta_g[0][1] + G_22 * theta_g[1][1];
    dv_xy_pos[idx].x = v_xy[idx].x * (0.5f * img_size.x);
    dv_xy_pos[idx].y = v_xy[idx].y * (0.5f * img_size.y);
}

/**
 * @brief .Initialize the Kernel function for backward projection of Gaussians
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
        radii,      //int* __restrict__ radii,
        conics,     // float3 __restrict__ conics
        // derivative from last layers
        dv_xy,
        dv_depth,
        dv_conic,
 * @returns mxArray *prhs[] parameters on the right-hand side; derivatives
        dv_cov2d, 
        dv_xy_pos, 
        dv_scales, 
        dv_rot
 */
void mexFunction(
    int nlhs, mxArray *plhs[],
    // dv_cov2d, 
    // dv_xy_pos, 
    // dv_scales, 
    // dv_rot
    int nrhs, mxArray const *prhs[]
    // mxGPUArray const *xy_pos;   // prhs[0]
    // mxGPUArray const *rot;      // prhs[1]
    // mxGPUArray const *scales;   // prhs[2]
    // mxGPUArray const *radii;    // prhs[3]
    // mxGPUArray const *conics;   // prhs[4]
    // // derivative from last layers
    // mxGPUArray const *dv_xy;    // prhs[5]
    // mxGPUArray const *dv_depth; // prhs[6]
    // mxGPUArray const *dv_conic; // prhs[7]
    // // image size               // prhs[8]
) {
    // input params 9
    mxGPUArray const *xy_pos;   // prhs[0]
    mxGPUArray const *rot;      // prhs[1]
    mxGPUArray const *scales;   // prhs[2]
    mxGPUArray const *radii;    // prhs[3]
    mxGPUArray const *conics;   // prhs[4]
    // derivative from last layers
    mxGPUArray const *dv_xy;    // prhs[5]
    mxGPUArray const *dv_depth; // prhs[6]
    mxGPUArray const *dv_conic; // prhs[7]
    // image size               // prhs[8]

    float2 const *d_xy_pos, *d_scales, *d_dv_xy;
    float3 const *d_conics, *d_dv_conic;
    float const *d_rot, *d_dv_depth;
    int const *d_radii;
    // prhs[3] is the image size [h,w]

    // output params 4
    mxGPUArray *dv_cov2d;
    mxGPUArray *dv_xy_pos;
    mxGPUArray *dv_scales;
    mxGPUArray *dv_rot;

    float2 *d_dv_xy_pos, *d_dv_scales;
    float3 *d_dv_cov2d;  // 3 X N conics params, N = number of Gaussian
    float *d_dv_rot;

    mxInitGPU();

    if (nrhs!=9) {
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input must have 9 inputs.");
    }

    if (mxIsGPUArray(prhs[8])){
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "The input #9 must not be a GPU array");
    }

    xy_pos  = mxGPUCreateFromMxArray(prhs[0]);
    rot     = mxGPUCreateFromMxArray(prhs[1]);
    scales  = mxGPUCreateFromMxArray(prhs[2]);
    radii   = mxGPUCreateFromMxArray(prhs[3]);
    conics  = mxGPUCreateFromMxArray(prhs[4]);
    dv_xy   = mxGPUCreateFromMxArray(prhs[5]);
    dv_depth  = mxGPUCreateFromMxArray(prhs[6]);
    dv_conic  = mxGPUCreateFromMxArray(prhs[7]);

    d_xy_pos    = (float2 const *)(mxGPUGetDataReadOnly(xy_pos));
    d_scales    = (float2 const *)(mxGPUGetDataReadOnly(scales));
    d_rot       = (float const *)(mxGPUGetDataReadOnly(rot));
    d_radii     = (int const *)(mxGPUGetDataReadOnly(radii));
    d_conics    = (float3 const *)(mxGPUGetDataReadOnly(conics));
    d_dv_xy     = (float2 const *)(mxGPUGetDataReadOnly(dv_xy));
    d_dv_depth  = (float const *)(mxGPUGetDataReadOnly(dv_depth));
    d_dv_conic  = (float3 const *)(mxGPUGetDataReadOnly(dv_conic));

    unsigned *img_size = (unsigned *) mxGetPr(prhs[8]); 
    unsigned img_h = img_size[0];
    unsigned img_w = img_size[1];
    
    const dim3 img_size_dim3(img_w, img_h, 1);

    // init output parameters
    dv_cov2d    = mxGPUCreateGPUArray(
                        mxGPUGetNumberOfDimensions(conics),
                        mxGPUGetDimensions(conics),
                        mxGPUGetClassID(conics),
                        mxGPUGetComplexity(conics),
                        MX_GPU_INITIALIZE_VALUES); 

    dv_xy_pos   = mxGPUCreateGPUArray(
                        mxGPUGetNumberOfDimensions(xy_pos),
                        mxGPUGetDimensions(xy_pos),
                        mxGPUGetClassID(xy_pos),
                        mxGPUGetComplexity(xy_pos),
                        MX_GPU_INITIALIZE_VALUES); 

    dv_scales   = mxGPUCreateGPUArray(
                        mxGPUGetNumberOfDimensions(scales),
                        mxGPUGetDimensions(scales),
                        mxGPUGetClassID(scales),
                        mxGPUGetComplexity(scales),
                        MX_GPU_INITIALIZE_VALUES); 

    dv_rot      = mxGPUCreateGPUArray(
                        mxGPUGetNumberOfDimensions(rot),
                        mxGPUGetDimensions(rot),
                        mxGPUGetClassID(rot),
                        mxGPUGetComplexity(rot),
                        MX_GPU_INITIALIZE_VALUES); 

    size_t num_points = mxGPUGetNumberOfElements(rot); // Number of Gaussian points

    d_dv_xy_pos  = (float2 *)(mxGPUGetData(dv_xy_pos));
    d_dv_scales  = (float2 *)(mxGPUGetData(dv_scales));
    d_dv_rot     = (float *)(mxGPUGetData(dv_rot));
    d_dv_cov2d   = (float3 *)(mxGPUGetData(dv_cov2d));


    const unsigned N_BLOCKS = (unsigned) ((unsigned) num_points + N_THREADS - 1) / N_THREADS;

    const dim3 tile_bounds = {
        (unsigned) ((img_w + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((img_h + BLOCK_Y - 1) / BLOCK_Y),
        1
    };

    
    project_2d_Gaussians_backward_kernel<<<N_BLOCKS, N_THREADS>>>(
        (int) num_points,  
        d_scales,       
        d_rot,          
        img_size_dim3,  
        d_radii,       
        d_conics,      
        d_dv_xy,        
        d_dv_depth,     
        d_dv_conic,     
        // output params
        d_dv_cov2d,    
        d_dv_xy_pos,   
        d_dv_scales,    
        d_dv_rot       
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(dv_cov2d);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dv_xy_pos);
    plhs[2] = mxGPUCreateMxArrayOnGPU(dv_rot);
    plhs[3] = mxGPUCreateMxArrayOnGPU(dv_scales);

    mxGPUDestroyGPUArray(xy_pos);
    mxGPUDestroyGPUArray(rot);
    mxGPUDestroyGPUArray(scales);
    mxGPUDestroyGPUArray(radii);
    mxGPUDestroyGPUArray(conics);
    mxGPUDestroyGPUArray(dv_xy);
    mxGPUDestroyGPUArray(dv_depth);
    mxGPUDestroyGPUArray(dv_conic);

    mxGPUDestroyGPUArray(dv_cov2d);
    mxGPUDestroyGPUArray(dv_xy_pos);
    mxGPUDestroyGPUArray(dv_scales);
    mxGPUDestroyGPUArray(dv_rot);
}