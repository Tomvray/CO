#ifndef UTILS_H
    #define UTILS_H

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include "parameters.h"

    // Functions

    // Soft-thresholding (shrinkage) operator (shrink(y, lambda * t))
    void shrink(double *y, double lambda_t, int n);

    // Matrix-vector multiplication: y = A * x
    void mat_vec_mult(double **A, double *x, double *y, int rows, int cols);

    //Compute norm of a vector
    double calculate_norm(double *x, int n);

    // Compute the objective function value: f(x) = ||Ax - b||^2 + lambda_1 * ||x||_1 + lambda_2/2 * ||x||^2
    double f(double **A, double *b, double *x, int rows, int cols);

    // Compute gradient: grad = A^T * (A*x - b) + lambda_2 * x
    double *compute_gradient(double **A, double *b, double *x, double *grad, int rows, int cols);

    //returns the step size with inexact line search backtracking
    double back_tracking_line_search(double **A, double *b, double *x, double *grad, int rows, int cols);


#endif