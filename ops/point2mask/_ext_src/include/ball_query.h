#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor pointsnum, const float radius, const int nsample);
