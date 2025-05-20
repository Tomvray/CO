#ifndef ALGORITHMS_H
    #define ALGORITHMS_H

    #include "utils.h"

    // Function prototypes

    // Iterative Shrinkage-Thresholding Algorithm (ISTA)
    double* ista( double *x, double **A, double *b, int rows, int cols);

    // Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    double* fista(double *x, double **A, double *b, int rows, int cols);

#endif