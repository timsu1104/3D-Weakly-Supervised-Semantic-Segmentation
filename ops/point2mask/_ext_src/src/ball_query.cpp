#include "ball_query.h"
#include "utils.h"

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_coords,
                                     const float *coords, int *idx, int *ptnum);

at::Tensor ball_query(at::Tensor new_coords, at::Tensor coords, at::Tensor pointsnum, const float radius, const int nsample) {
  CHECK_CONTIGUOUS(new_coords);
  CHECK_CONTIGUOUS(coords);
  CHECK_CONTIGUOUS(pointsnum);
  CHECK_IS_FLOAT(new_coords);
  CHECK_IS_FLOAT(coords);
  CHECK_IS_INT(pointsnum);

  if (new_coords.is_cuda()) {
    CHECK_CUDA(coords);
  }

  at::Tensor idx =
      -torch::ones({new_coords.size(0), new_coords.size(1), nsample},
                   at::device(new_coords.device()).dtype(at::ScalarType::Int)); // B, npoints, nsample; -1 means empty

  if (new_coords.is_cuda()) {
    query_ball_point_kernel_wrapper(coords.size(0), coords.size(1), new_coords.size(1),
                                    radius, nsample, new_coords.data_ptr<float>(),
                                    coords.data_ptr<float>(), idx.data_ptr<int>(), pointsnum.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}
