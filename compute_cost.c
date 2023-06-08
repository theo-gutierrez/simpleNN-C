#include <math.h>

#include "structures.h"

double compute_cost(double **A2, double **Y, struct Sizes sizes) {
    double cost = 0.0;
    int m = sizes.m;

    for (int i = 0; i < m; i++) {
        cost += Y[0][i] * log(A2[0][i]) + (1 - Y[0][i]) * log(1 - A2[0][i]);
    }

    cost *= (-1.0 / m);

    return cost;
}