#include <stdio.h>
#include <stdlib.h>
#include "initialize_parameters.h"

int main() {
    int n_x = 2; // n_x -- size of the input layer
    int n_h = 6; // n_h -- size of the hidden layer
    int n_y = 2; // n_y -- size of the output layer

    struct Parameters parameters = initialize_parameters(n_x, n_h, n_y);

    // Access the parameters as parameters.W1, parameters.b1, parameters.W2, parameters.b2

    printf("W1:\n");
    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", parameters.W1[i][j]);
        }
        printf("\n");
    }

    printf("b1:\n");
    for (int i = 0; i < n_h; i++) {
        printf("%f\n", parameters.b1[i][0]);
    }

    printf("W2:\n");
    for (int i = 0; i < n_y; i++) {
        for (int j = 0; j < n_h; j++) {
            printf("%f ", parameters.W2[i][j]);
        }
        printf("\n");
    }

    printf("b2:\n");
    for (int i = 0; i < n_y; i++) {
        printf("%f\n", parameters.b2[i][0]);
    }

    // Free the allocated memory
    for (int i = 0; i < n_h; i++) {
        free(parameters.W1[i]);
    }
    free(parameters.W1);

    for (int i = 0; i < n_h; i++) {
        free(parameters.b1[i]);
    }
    free(parameters.b1);

    for (int i = 0; i < n_y; i++) {
        free(parameters.W2[i]);
    }
    free(parameters.W2);

    for (int i = 0; i < n_y; i++) {
        free(parameters.b2[i]);
    }
    free(parameters.b2);

    return 0;
}
