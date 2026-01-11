#include "dataset.hpp"
#include <random>

Eigen::MatrixXd generate_blobs(
    int samples,
    int features,
    int clusters,
    double spread
) {
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, spread);
    std::uniform_real_distribution<double> center_dist(-10.0, 10.0);

    Eigen::MatrixXd centers(clusters, features);
    for (int i = 0; i < clusters; ++i)
        for (int j = 0; j < features; ++j)
            centers(i, j) = center_dist(rng);

    Eigen::MatrixXd data(samples, features);

    for (int i = 0; i < samples; ++i) {
        int c = i % clusters;
        for (int j = 0; j < features; ++j)
            data(i, j) = centers(c, j) + noise(rng);
    }

    return data;
}
