#ifndef STRUCTURES_H
#define STRUCTURES_H

struct Cache {
    double** Z1;
    double** A1;
    double** Z2;
    double** A2;
};

struct Parameters {
    double** W1;
    double** b1;
    double** W2;
    double** b2;
};

struct Sizes {
    int n_x;
    int n_h;
    int n_y;
    int m;
};

struct Grads {
    double** dW1;
    double** db1;
    double** dW2;
    double** db2;   
};

#endif
