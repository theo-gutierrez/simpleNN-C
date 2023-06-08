#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structures.h"
#include "forward_propagation.h"

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

struct Cache forward_propagation(double** X, struct Parameters parameters, struct Sizes sizes) {
    struct Cache cache;
    
    double** W1 = parameters.W1;
    double** b1 = parameters.b1;
    double** W2 = parameters.W2;
    double** b2 = parameters.b2;
    
    // Get the shapes
    int n_x = sizes.n_x;
    int n_h = sizes.n_h;
    int n_y = sizes.n_y;
    int m = sizes.m;

    // Compute Z1
    cache.Z1 = (double**)malloc(n_h * sizeof(double*));
    for (int i = 0; i < n_h; i++) {
        cache.Z1[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            cache.Z1[i][j] = 0.0;
            for (int k = 0; k < n_x; k++) {
                cache.Z1[i][j] += W1[i][k] * X[k][j];
            }
            cache.Z1[i][j] += b1[i][0];
        }
    }
    
    // Compute A1 using tanh activation function
    cache.A1 = (double**)malloc(n_h * sizeof(double*));
    for (int i = 0; i < n_h; i++) {
        cache.A1[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            cache.A1[i][j] = tanh(cache.Z1[i][j]);
        }
    }
    
    // Compute Z2
    cache.Z2 = (double**)malloc(n_y * sizeof(double*));
    for (int i = 0; i < n_y; i++) {
        cache.Z2[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            cache.Z2[i][j] = 0.0;
            for (int k = 0; k < n_h; k++) {
                cache.Z2[i][j] += W2[i][k] * cache.A1[k][j];
            }
            cache.Z2[i][j] += b2[i][0];
        }
    }
    
    // Compute A2 (sigmoid output)
    cache.A2 = (double**)malloc(n_y * sizeof(double*));
    for (int i = 0; i < n_y; i++) {
        cache.A2[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            cache.A2[i][j] = sigmoid(cache.Z2[i][j]);
        }
    }
    
    return cache;
}
