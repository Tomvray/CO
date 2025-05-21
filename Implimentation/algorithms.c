#include "algorithms.h"

// ISTA Algorithm
double* ista(double *x, double **A, double *b, int rows, int cols) {

    // double t_k = 1.0 / compute_lipschitz(A, rows, cols); // Lipschitz constant
    double t_k = T_0;
    double *grad = (double *)malloc(cols * sizeof(double));
    double *x_new = (double *)malloc(cols * sizeof(double));

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // compute gradient g(x_k) = A^T(Ax_k - b)
        compute_gradient(A, b, x, grad, rows, cols);

        // compute new t_k
        t_k = back_tracking_line_search(A, b, x, grad, rows, cols); 

        //printf("t_k ista: %f gradnorm: %f\n", t_k, calculate_norm(grad, cols));

        // x_new = x - t_k * grad g(x_k)
        for (int i = 0; i < cols; i++) {
            x_new[i] = x[i] - t_k * grad[i];
        }
        //shrink (x_k - t_k * grad; LAMBDA_1 * t_k)
        shrink(x_new, LAMBDA_1 * t_k, cols);

        // Check convergence ||grad||_2 <= TOL
        if (calculate_norm(grad, cols) < TOL) {
            printf("ISTA Converged at iteration %d\n", iter);
            break;
        }

        // Update x
        for (int i = 0; i < cols; i++) {
            x[i] = x_new[i];
        }

        //printf("objective function value: %f\n", f(A, b, x, rows, cols));
    }

    free(grad);
    free(x_new);

    return x;
}

// FISTA Algorithm
double* fista(double *x, double **A, double *b, int rows, int cols){
    double t_k = 1;
    double momentum_factor = 1.0;
    double momentum_factor_new;

    double *grad = (double *)malloc(cols * sizeof(double));
    double *x_new = (double *)malloc(cols * sizeof(double));
    double *y = (double *)malloc(cols * sizeof(double));
    if (grad == NULL || x_new == NULL || y == NULL) {
        printf("Error: Memory allocation failed\n");
        return NULL;
    }
    // Initialize y
    for (int i = 0; i < cols; i++) y[i] = x[i];

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // compute gradient g(y_k) = A^T(Ay_k - b)
        compute_gradient(A, b, y, grad, rows, cols);

        // compute new t_k
        t_k = back_tracking_line_search(A, b, x, grad, rows, cols);

        //printf("t_k: %f\n", t_k);

        // x_new = y - t_k * grad g(y_k)
        for (int i = 0; i < cols; i++)
            x_new[i] = y[i] - t_k * grad[i];
        
        //shrink (x_k - t_k * grad; LAMBDA_1 * t_k)
        shrink(x_new, LAMBDA_1 * t_k, cols);

        //Update momentum factor
        momentum_factor_new = (1.0 + sqrt(1.0 + 4.0 * momentum_factor* momentum_factor)) / 2.0;
        if (momentum_factor_new < 0) {
            printf("Error: momentum_factor_new < 0\n");
            free(grad);
            free(x_new);
            free(y);
            return NULL;
        }
        
        // Update y = x_new + ((momentum_factor - 1) / momentum_factor_new) * (x_new - x)
        for (int i = 0; i < cols; i++) y[i] = x_new[i] + ((momentum_factor - 1) / momentum_factor_new) * (x_new[i] - x[i]);
        
        // Update x
        for (int i = 0; i < cols; i++) x[i] = x_new[i];
        
        
        // Update momentum factor
        momentum_factor = momentum_factor_new;

        // Check convergence ||grad||_2 <= TOL
        // printf("Convergence check: %f\n", calculate_norm(grad, cols));
        if (calculate_norm(grad, cols) < TOL) {
            printf("FISTA converged at iteration %d\n", iter);
            break;
        }

        //printf("objective function value: %f\n", f(A, b, x, rows, cols));

    }

    free(grad);
    free(x_new);
    free(y);


    return x;
}