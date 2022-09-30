#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_coords(b, m, 2) coords(b, n, 2)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_coords,
                                        const float *__restrict__ coords,
                                        int *__restrict__ idx,
                                        int *__restrict__ pointnums) {
  int batch_index = blockIdx.x;
  coords += batch_index * n * 2;
  new_coords += batch_index * m * 2;
  idx += m * nsample * batch_index;
  int ptnum = pointnums[batch_index];

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_coords[j << 1];
    float new_y = new_coords[j << 1 | 1];
    for (int k = 0, cnt = 0; k < n - ptnum && cnt < nsample; ++k) {
      float x = coords[k << 1];
      float y = coords[k << 1 | 1];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y);
    //   printf("Checking (%f, %f) for (%f, %f).\n", x, y, new_x, new_y);
      if (d2 < radius2) {
        // printf("(%f, %f) for (%f, %f) is chosen. distance is %f, less than %f\n", x, y, new_x, new_y, d2, radius2);
        // if (cnt == 0) {
        //   for (int l = 0; l < nsample; ++l) {
        //     idx[j * nsample + l] = k;
        //   }
        // }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_coords,
                                     const float *coords, int *idx, int *ptnum) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_coords, coords, idx, ptnum);

  CUDA_CHECK_ERRORS();
}
