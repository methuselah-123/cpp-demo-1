#pragma once
#include <Eigen/Dense>
#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iters = 100);

    void fit(const Eigen::MatrixXd& X);
    std::vector<int> predict(const Eigen::MatrixXd& X) const;

    const Eigen::MatrixXd& centroids() const;

private:
    int k;
    int max_iters;
    Eigen::MatrixXd centroids_;

    int closest_centroid(const Eigen::VectorXd& x) const;
};
