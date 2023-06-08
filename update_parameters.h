#ifndef UPDATE_PARAMETERS_H
#define UPDATE_PARAMETERS_H

#include "structures.h"

struct Parameters update_parameters(struct Parameters parameters, struct Grads grads, double learning_rate, struct Sizes sizes);

#endif