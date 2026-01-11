#include "kmeans.hpp"
#include <random>
#include <limits>

KMeans::KMeans(int k, int max_iters)
    : k(k), max_iters(max_iters) {}

int KMeans::closest_centroid(const Eigen::VectorXd& x) const {
    double best_dist = std::numeric_limits<double>::max();
    int best_idx = 0;

    for (int i = 0; i < k; ++i) {
        double dist = (x - centroids_.row(i).transpose()).squaredNorm();
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

void KMeans::fit(const Eigen::MatrixXd& X) {
    int n = X.rows();
    int d = X.cols();

    centroids_.resize(k, d);

    // Random initialization
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < k; ++i)
        centroids_.row(i) = X.row(dist(rng));

    std::vector<int> labels(n);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assignment step
        for (int i = 0; i < n; ++i)
            labels[i] = closest_centroid(X.row(i).transpose());

        // Update step
        centroids_.setZero();
        Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);

        for (int i = 0; i < n; ++i) {
            centroids_.row(labels[i]) += X.row(i);
            counts(labels[i])++;
        }

        for (int i = 0; i < k; ++i)
            if (counts(i) > 0)
                centroids_.row(i) /= counts(i);
    }
}

std::vector<int> KMeans::predict(const Eigen::MatrixXd& X) const {
    int n = X.rows();
    std::vector<int> labels(n);

    for (int i = 0; i < n; ++i)
        labels[i] = closest_centroid(X.row(i).transpose());

    return labels;
}

const Eigen::MatrixXd& KMeans::centroids() const {
    return centroids_;
}
