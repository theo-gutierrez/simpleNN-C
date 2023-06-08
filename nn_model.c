#include <stdio.h>
#include "nn_model.h"
#include "initialize_parameters.h"
#include "forward_propagation.h"
#include "compute_cost.h"
#include "backward_propagation.h"
#include "update_parameters.h"


struct Parameters nn_model(double** X, double** Y, int num_iterations, int print_cost, struct Sizes sizes) {

    int n_x = sizes.n_x;
    int n_h = sizes.n_h;
    int n_y = sizes.n_y;

    struct Parameters parameters; 
    struct Cache cache;
    double cost;
    struct Grads grads;
    
    parameters = initialize_parameters(n_x, n_h, n_y);
    
    for (int i = 0; i < num_iterations ; i++) {
        // Forward propagation. Inputs: "X, parameters". Outputs: "cache".
        cache = forward_propagation(X, parameters, sizes);
        
        // Cost function. Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(cache.A2, Y, sizes);
 
        // Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y, sizes);
 
        // Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, 1.2, sizes);

        if (print_cost == 1 && i % 100 == 0) {
            printf("Cost after iteration %d: %f\n", i, cost);
        }
    }
    return parameters;
}
