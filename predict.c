#include <stdio.h>
#include <stdlib.h>
#include "structures.h"
#include "forward_propagation.h"

int* predict(struct Parameters parameters, double** X, struct Sizes sizes) {
    int* predictions = malloc(sizes.m * sizeof(int));
    struct Cache cache;
    cache = forward_propagation(X, parameters, sizes);
    for (int i = 0; i < sizes.m; i++) {
        printf("probability %f\n", cache.A2[0][i]);
        if (cache.A2[0][i] > 0.5) {
            predictions[i] = 1;
        } else {
            predictions[i] = 0;
        }
    }
    return predictions;
}