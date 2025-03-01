// Authors: Shuhe Zhang, Yating Chen, Liangcai Cao
// Tsinghua University
// shuhe-zhang@tsinghua.edu.cn, clc@tsinghua.edu.cn
//

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define N_THREADS 256 // maximum thread is 1024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities

#define MAX_REGISTER_CHANNELS 3
