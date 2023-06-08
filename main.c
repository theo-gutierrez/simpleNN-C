#include <stdio.h>
#include <stdlib.h>
#include "initialize_parameters.h"
#include "forward_propagation.h"
#include "structures.h"
#include "compute_cost.h"
#include "backward_propagation.h"
#include "update_parameters.h"
#include "nn_model.h"
#include "predict.h"

int main() {

    double **X = malloc(4 * sizeof(double *));
    double **Y = malloc(4 * sizeof(double *));

    double values[4][3] = {{3.0, 4.0, 2.0},
                           {5.0, 3.0, 1.0},
                           {2.0, 9.0, 8.0},
                           {1.0, 5.0, 6.0}};

    double values_y[4][1] = {{0.0},
                             {1.0},
                             {1.0},
                             {0.0}};

    for (int i = 0; i < 4; i++) {
        X[i] = malloc(3 * sizeof(double));
        for (int j = 0; j < 3; j++) {
            X[i][j] = values[i][j];
        }
    }

    for (int i = 0; i < 4; i++) {
        Y[i] = malloc(sizeof(double));
        for (int j = 0; j < 1; j++) {
            Y[i][j] = values_y[i][j];
        }
    }

    int n_x = 3;
    int n_h = 6;
    int n_y = 1;
    int m = 4;

    double learning_rate = 1.2; 

    struct Sizes sizes;
    sizes.n_x = n_x; // n_x -- size of the input layer
    sizes.n_h = n_h; // n_h -- size of the hidden layer
    sizes.n_y = n_y; // n_y -- size of the output layer
    sizes.m = m; 

    int* predictions = malloc(m * sizeof(int));


    struct Parameters parameters = initialize_parameters(sizes.n_x, sizes.n_h, sizes.n_y);

    struct Cache cache = forward_propagation(X, parameters, sizes);

    double cost = compute_cost(cache.A2, Y, sizes);

    struct Grads grads = backward_propagation(parameters,  cache, X, Y, sizes);

    struct Parameters updated_parameters = update_parameters(parameters, grads, learning_rate, sizes);

    struct Parameters final_parameters = nn_model(X, Y, 1000, 1, sizes);

    predictions = predict(final_parameters, X, sizes);  

    printf("X:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", X[i][j]);
        }
        printf("\n");
    }    

    printf("X:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", X[i][j]);
        }
        printf("\n");
    }    


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

    printf("cache:\n");
    printf("Z1:\n");
    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", cache.Z1[i][j]);
        }
        printf("\n");
    }

    printf("A1:\n");
    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", cache.A1[i][j]);
        }
        printf("\n");
    }

    printf("Z2:\n");
    for (int i = 0; i < n_y; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", cache.Z2[i][j]);
        }
        printf("\n");
    }

    printf("A2:\n");
    for (int i = 0; i < n_y; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", cache.A2[i][j]);
        }
        printf("\n");
    }

    printf("Cost:\n");
    printf("%f \n", cost);

    printf("dW1:\n");
    for (int i = 0; i < sizes.n_h; i++) {
        for (int j = 0; j < sizes.n_x; j++) {
            printf("%f ", grads.dW1[i][j]);
        }
        printf("\n");
    }
    
    printf("db1:\n");
    for (int i = 0; i < sizes.n_h; i++) {
        printf("%f\n", grads.db1[i][0]);
    }
    
    printf("dW2:\n");
    for (int i = 0; i < sizes.n_y; i++) {
        for (int j = 0; j < sizes.n_h; j++) {
            printf("%f ", grads.dW2[i][j]);
        }
        printf("\n");
    }
    
    printf("db2:\n");
    for (int i = 0; i < sizes.n_y; i++) {
        printf("%f\n", grads.db2[i][0]);
    }
    printf("updated_parameters:\n");
    printf("W1:\n");
    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_x; j++) {
            printf("%f ", updated_parameters.W1[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("b1:\n");
    for (int i = 0; i < n_h; i++) {
        printf("%f\n", updated_parameters.b1[i][0]);
    }
    printf("\n");

    printf("W2:\n");
    for (int i = 0; i < n_y; i++) {
        for (int j = 0; j < n_h; j++) {
            printf("%f ", updated_parameters.W2[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("b2:\n");
    for (int i = 0; i < n_y; i++) {
        printf("%f\n", updated_parameters.b2[i][0]);
    }
    printf("\n");

    printf("Predictions:\n");
    for (int i = 0; i < m; i++) {
        printf("%d", predictions[i]);
        printf("\n");
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

    // Free cache.A2
    for (int i = 0; i < n_y; i++) {
        free(cache.A2[i]);
    }
    free(cache.A2);

    // Free cache.Z2
    for (int i = 0; i < n_y; i++) {
        free(cache.Z2[i]);
    }
    free(cache.Z2);

    // Free cache.A1
    for (int i = 0; i < n_h; i++) {
        free(cache.A1[i]);
    }
    free(cache.A1);

    // Free cache.Z1
    for (int i = 0; i < n_h; i++) {
        free(cache.Z1[i]);
    }
    free(cache.Z1);

    for (int i = 0; i < sizes.n_h; i++) {
        free(grads.dW1[i]);
        free(grads.db1[i]);
    }

    for (int i = 0; i < sizes.n_y; i++) {
        free(grads.dW2[i]);
        free(grads.db2[i]);
    }

    free(grads.dW1);
    free(grads.db1);
    free(grads.dW2);
    free(grads.db2);

    return 0;
}
