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
        /*if (calculate_norm(grad, cols) < TOL) {
            printf("ISTA Converged at iteration %d\n", iter);
            break;
        }*/

        // Check convergence f(x_k) - f(x_new) < TOL
        if (fabs(f(A, b, x, rows, cols) - f(A, b, x_new, rows, cols)) < TOL) {
            printf("ISTA converged at iteration %d\n", iter);
            break;
        }

        // Update x
        for (int i = 0; i < cols; i++) {
            x[i] = x_new[i];
        }

        //printf("objective function value: %f\n", f(A, b, x, rows, cols));
    }

    printf("ISTA converged at iteration with value of f : %f\n", f(A, b, x, rows, cols));
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
        
        // Check convergence ||grad||_2 <= TOL
        // printf("Convergence check: %f\n", calculate_norm(grad, cols));
        if (fabs(f(A, b, x, rows, cols) - f(A, b, x_new, rows, cols)) < TOL) {
            printf("FISTA converged at iteration %d\n", iter);
            break;
        }
        // Update x
        for (int i = 0; i < cols; i++) x[i] = x_new[i];
        
        // Update momentum factor
        momentum_factor = momentum_factor_new;

        /*if (calculate_norm(grad, cols) < TOL) {
            printf("FISTA converged at iteration %d\n", iter);
            break;
        }*/

        //printf("objective function value: %f\n", f(A, b, x, rows, cols));

    }

    printf("FISTA converged at iteration with value of f :%f\n", f(A, b, x, rows, cols));
    free(grad);
    free(x_new);
    free(y);
    return x;
}

//typedef double (*objective_func)(double **A, double *b, double *x, int rows, int cols);
//Compute the proximal gradiant using L-BFGS




// Example gradient of the proximal objective: grad = x - v + lambda * grad_g(x)
void prox_objective_grad(double *x, double *v, double lambda, int n, void (*grad_g)(double*, double*, int), double *grad) {
    double *grad_gx = (double*)malloc(n * sizeof(double));
    grad_g(x, grad_gx, n); // User-supplied gradient of g at x
    for (int i = 0; i < n; i++) {
        grad[i] = x[i] - v[i] + lambda * grad_gx[i];
    }
    free(grad_gx);
}

// Proximal operator using BFGS (simplified, not full BFGS implementation)
double* prox_bfgs(double *v, double lambda, int n, void (*grad_g)(double*, double*, int), int max_iter, double tol) {
    double *x = (double*)malloc(n * sizeof(double));
    double *grad = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) x[i] = v[i]; // Initialize x = v

    for (int iter = 0; iter < max_iter; iter++) {
        prox_objective_grad(x, v, lambda, n, grad_g, grad);

        // Simple gradient descent step (replace with BFGS update for real use)
        double alpha = 1e-2; // Step size, should use line search or BFGS update
        for (int i = 0; i < n; i++) x[i] -= alpha * grad[i];

        // Check convergence
        double norm = 0.0;
        for (int i = 0; i < n; i++) norm += grad[i] * grad[i];
        if (sqrt(norm) < tol) break;
    }
    free(grad);
    return x;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Function prototypes
typedef double (*objective_func)(double*);
typedef void (*grad_func)(double*, double*);

// Dot product function
double dot_product(double* v1, double* v2, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Copy vector function
void copy_vector(double* dest, double* src, int n) {
    memcpy(dest, src, n * sizeof(double));
}

// L-BFGS implementation
double* L_BFGS(double* x, int dimensions, int m, objective_func obj, grad_func grad, int max_iter, double tol) {
    double alpha[m], rho[m];
    double *q = malloc(dimensions * sizeof(double));
    double *grad_new = malloc(dimensions * sizeof(double));
    double *s[m], *y[m];

    for (int i = 0; i < m; i++) {
        s[i] = malloc(dimensions * sizeof(double));
        y[i] = malloc(dimensions * sizeof(double));
    }

    double *grad_old = malloc(dimensions * sizeof(double));
    double *x_old = malloc(dimensions * sizeof(double));

    grad(x, grad_old);
    int k = 0;

    while (k < max_iter && sqrt(dot_product(grad_old, grad_old, dimensions)) > tol) {
        copy_vector(q, grad_old, dimensions);

        int upper_bound = (k < m) ? k : m;
        for (int i = upper_bound - 1; i >= 0; i--) {
            rho[i] = 1.0 / dot_product(y[i], s[i], dimensions);
            alpha[i] = rho[i] * dot_product(s[i], q, dimensions);
            for (int j = 0; j < dimensions; j++)
                q[j] -= alpha[i] * y[i][j];
        }

        double gamma = 1.0;
        if (k > 0) {
            gamma = dot_product(s[upper_bound - 1], y[upper_bound - 1], dimensions) /
                    dot_product(y[upper_bound - 1], y[upper_bound - 1], dimensions);
        }

        for (int i = 0; i < dimensions; i++)
            q[i] *= gamma;

        for (int i = 0; i < upper_bound; i++) {
            double beta = rho[i] * dot_product(y[i], q, dimensions);
            for (int j = 0; j < dimensions; j++)
                q[j] += s[i][j] * (alpha[i] - beta);
        }

        for (int i = 0; i < dimensions; i++) {
            x_old[i] = x[i];
            x[i] -= q[i];
        }

        grad(x, grad_new);

        if (k >= m) {
            free(s[0]);
            free(y[0]);
            memmove(s, s+1, (m-1)*sizeof(double*));
            memmove(y, y+1, (m-1)*sizeof(double*));
            s[m-1] = malloc(dimensions * sizeof(double));
            y[m-1] = malloc(dimensions * sizeof(double));
        }

        for (int i = 0; i < dimensions; i++) {
            s[upper_bound - 1][i] = x[i] - x_old[i];
            y[upper_bound - 1][i] = grad_new[i] - grad_old[i];
        }

        copy_vector(grad_old, grad_new, dimensions);

        k++;
    }

    free(q);
    free(grad_new);
    free(grad_old);
    free(x_old);
    for (int i = 0; i < m; i++) {
        free(s[i]);
        free(y[i]);
    }

    return x;
}