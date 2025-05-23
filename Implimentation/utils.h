#ifndef UTILS_H
    #define UTILS_H

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <string.h>
    #include <time.h>
    #include "parameters.h"

    typedef struct Data Data;
    
    typedef double (*objective_func)(double*, Data);
    typedef double* (*grad_func)(double*, double*, Data);

    // Function prototypes
    typedef struct Data {
        double **A;
        double *b;
        int rows;
        int cols;
        double t_0; // Initial step size
        objective_func prox_func; // Only used in the LBFGS approximation of the prox operator (h(x))
        grad_func prox_grad; // Only used in the LBFGS approximation of the prox operator (grad(h(x)))
        double *x; // Point arround which we are computing the prox operator
    }   Data;

    typedef struct Problem {
        Data data;
        objective_func objective_func;
        grad_func grad_func;
    }   Problem;

    // Soft-thresholding (shrinkage) operator (shrink(y, lambda * t))
    void shrink(double *y, double lambda_t, int n);

    // Compute data.prox_function(u) + 1/2 * ||u - Problem.x||^2
    // compute h(u) + 1/2 * ||u - x||^2
    double prox_function(double *u, Data data);

    // Compute the gradient of the proximal function data.grad_prox_function(u) + (u - data.x)
    double* prox_gradient(double *u, double* grad, Data data);

    // l2
    double l2_func (double* x, Data data);

    // gradient of l2
    double* l2_grad (double* x, double* grad, Data data);

    // Matrix-vector multiplication: y = A * x
    void mat_vec_mult(double **A, double *x, double *y, int rows, int cols);

    //Compute norm of a vector
    double calculate_norm(double *x, int n);

    // Dot product function
    double dot_product(double* v1, double* v2, int n);

    // Compute the objective function value: f(x) = ||Ax - b||^2 + lambda_1 * ||x||_1 + lambda_2/2 * ||x||^2
    double f(double **A, double *b, double *x, int rows, int cols);

    // Compute gradient: grad = A^T * (A*x - b) + lambda_2 * x
    double *compute_gradient(double **A, double *b, double *x, double *grad, int rows, int cols);

    //returns the step size with inexact line search backtracking
    double back_tracking_line_search(double *x, double *grad, Data data);

    double power_iteration(double **A, int rows, int cols, int it_max, double tol);

    double estimate_t0(double **A, int rows, int cols);

    double* grad_LS(double *x, double *grad, Data data);

#endif