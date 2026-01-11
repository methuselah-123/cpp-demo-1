#include <iostream>
#include <chrono>
#include "dataset.hpp"
#include "kmeans.hpp"

int main() {
    int samples = 10000;
    int features = 2;
    int clusters = 4;

    auto X = generate_blobs(samples, features, clusters, 0.8);

    KMeans model(clusters, 100);

    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "K-Means completed in "
              << elapsed.count() << " seconds\n";

    std::cout << "Centroids:\n"
              << model.centroids() << "\n";

    return 0;
}
