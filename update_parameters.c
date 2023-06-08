#include "structures.h"

struct Parameters update_parameters(struct Parameters parameters, struct Grads grads, double learning_rate, struct Sizes sizes) {
    struct Parameters updated_parameters;

    double** W1 = parameters.W1;
    double** b1 = parameters.b1;
    double** W2 = parameters.W2;
    double** b2 = parameters.b2;

    double** dW1 = grads.dW1;
    double** db1 = grads.db1;
    double** dW2 = grads.dW2;
    double** db2 = grads.db2;

    int n_x = sizes.n_x;
    int n_h = sizes.n_h;
    int n_y = sizes.n_y;
    
    // Update rule for W1
    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_x; j++) {
            W1[i][j] -= learning_rate * dW1[i][j];
        }
    }
    
    // Update rule for b1
    for (int i = 0; i < n_h; i++) {
        b1[i][0] -= learning_rate * db1[i][0];
    }
    
    // Update rule for W2
    for (int i = 0; i < n_y; i++) {
        for (int j = 0; j < n_h; j++) {
            W2[i][j] -= learning_rate * dW2[i][j];
        }
    }
    
    // Update rule for b2
    for (int i = 0; i < n_y; i++) {
        b2[i][0] -= learning_rate * db2[i][0];
    }
    
    updated_parameters.W1 = W1;
    updated_parameters.b1 = b1;
    updated_parameters.W2 = W2;
    updated_parameters.b2 = b2;
    
    return updated_parameters;
}
