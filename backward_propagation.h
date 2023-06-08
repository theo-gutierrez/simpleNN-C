#ifndef BACKWARD_PROPAGATION_H
#define BACKWARD_PROPAGATION_H

#include "structures.h"

struct Grads backward_propagation(struct Parameters parameters, struct Cache cache, double** X, double** Y, struct Sizes sizes);

#endif