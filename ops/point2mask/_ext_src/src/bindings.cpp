#include "ball_query.h"
#include "group_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &ball_query);

  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
}
