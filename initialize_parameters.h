#ifndef INITIALIZE_PARAMETERS_H
#define INITIALIZE_PARAMETERS_H

struct Parameters {
    double** W1;
    double** b1;
    double** W2;
    double** b2;
};

struct Parameters initialize_parameters(int n_x, int n_h, int n_y);

#endif /* INITIALIZE_PARAMETERS_H */
