#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "structures.h"

struct Parameters initialize_parameters(int n_x, int n_h, int n_y) {
    struct Parameters params;

    // Allocate memory for W1
    params.W1 = (double**)malloc(n_h * sizeof(double*));
    for (int i = 0; i < n_h; i++) {
        params.W1[i] = (double*)malloc(n_x * sizeof(double));
    }

    // Allocate memory for b1
    params.b1 = (double**)malloc(n_h * sizeof(double*));
    for (int i = 0; i < n_h; i++) {
        params.b1[i] = (double*)malloc(sizeof(double));
    }

    // Allocate memory for W2
    params.W2 = (double**)malloc(n_y * sizeof(double*));
    for (int i = 0; i < n_y; i++) {
        params.W2[i] = (double*)malloc(n_h * sizeof(double));
    }

    // Allocate memory for b2
    params.b2 = (double**)malloc(n_y * sizeof(double*));
    for (int i = 0; i < n_y; i++) {
        params.b2[i] = (double*)malloc(sizeof(double));
    }

    // Initialize W1 with random values
    srand(time(NULL));
    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_x; j++) {
            double random_value = (double)rand() / RAND_MAX;
            params.W1[i][j] = random_value * 0.01;
        }
    }

    // Initialize b1 with zeros
    for (int i = 0; i < n_h; i++) {
        params.b1[i][0] = 0.0;
    }

    // Initialize W2 with random values
    for (int i = 0; i < n_y; i++) {
        for (int j = 0; j < n_h; j++) {
            double random_value = (double)rand() / RAND_MAX;
            params.W2[i][j] = random_value * 0.01;
        }
    }

    // Initialize b2 with zeros
    for (int i = 0; i < n_y; i++) {
        params.b2[i][0] = 0.0;
    }

    return params;
}
