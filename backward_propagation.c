#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structures.h"
#include "backward_propagation.h"

struct Grads backward_propagation(struct Parameters parameters, struct Cache cache, double** X, double** Y, struct Sizes sizes) {
    struct Grads grads;

    double** dZ2;

    double** W1 = parameters.W1;
    double** W2 = parameters.W2;

    double** A1 = cache.A1;
    double** A2 = cache.A2;

    int n_y = sizes.n_y;
    int m = sizes.m;
    int n_h = sizes.n_h;

    dZ2 = (double**)malloc(n_y * sizeof(double*));
    for (int i = 0; i < n_y; i++) {
        dZ2[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            dZ2[i][j] = A2[i][j] - Y[0][j];
        }
    }

    grads.dW2 = (double**)malloc(sizes.n_y * sizeof(double*));
    for (int i = 0; i < sizes.n_y; i++) {
        grads.dW2[i] = (double*)malloc(sizes.n_h * sizeof(double));
        for (int j = 0; j < sizes.n_h; j++) {
            grads.dW2[i][j] = 0.0;
            for (int k = 0; k < m; k++) {
                grads.dW2[i][j] += dZ2[i][k] * A1[j][k];
            }
            grads.dW2[i][j] *= (1.0 / m);
        }
    }
    grads.db2 = (double**)malloc(sizes.n_y * sizeof(double*));
    for (int i = 0; i < sizes.n_y; i++) {
        grads.db2[i] = (double*)malloc(sizeof(double));
        grads.db2[i][0] = 0.0;
        for (int j = 0; j < m; j++) {
            grads.db2[i][0] += (1.0 / m) * dZ2[i][j];
        }
    }
    
    double** dZ1 = (double**)malloc(sizes.n_h * sizeof(double*));
    for (int i = 0; i < sizes.n_h; i++) {
        dZ1[i] = (double*)malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            dZ1[i][j] = 0.0;
            for (int k = 0; k < sizes.n_y; k++) {
                dZ1[i][j] += W2[k][i] * dZ2[k][j] * (1 - pow(A1[i][j], 2));
            }
        }
    }
    
    grads.dW1 = (double**)malloc(sizes.n_h * sizeof(double*));
    for (int i = 0; i < sizes.n_h; i++) {
        grads.dW1[i] = (double*)malloc(sizes.n_x * sizeof(double));
        for (int j = 0; j < sizes.n_x; j++) {
            grads.dW1[i][j] = 0.0;
            for (int k = 0; k < m; k++) {
                grads.dW1[i][j] += dZ1[i][k] * X[j][k];
            }
            grads.dW1[i][j] *= (1.0 / m);
        }
    }
    
    grads.db1 = (double**)malloc(sizes.n_h * sizeof(double*));
    for (int i = 0; i < sizes.n_h; i++) {
        grads.db1[i] = (double*)malloc(sizeof(double));
        grads.db1[i][0] = 0.0;
        for (int j = 0; j < m; j++) {
            grads.db1[i][0] += (1.0 / m) * dZ1[i][j];
        }
    }
    
    return grads;

}