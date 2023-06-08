#ifndef NN_MODEL_H
#define NN_MODEL_H

#include "structures.h"

struct Parameters nn_model(double** X, double** Y, int num_iterations, int print_cost, struct Sizes sizes);


#endif