#ifndef ALGORITHMS_H
    #define ALGORITHMS_H

    #include "utils.h"

    // Function prototypes

    // Iterative Shrinkage-Thresholding Algorithm (ISTA)
    double* ista(double *x, Problem problem, FILE *file);

    // Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    double* fista(double *x, Problem problem, FILE *file);

    // L-BFGS implementation
    double* L_BFGS(double* x, int m, Problem problem, FILE *file);

    
    double* LBFGS_fista(double *x, int m, Problem problem, FILE *file);

#endif