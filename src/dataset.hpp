#pragma once
#include <Eigen/Dense>

Eigen::MatrixXd generate_blobs(
    int samples,
    int features,
    int clusters,
    double spread
);
