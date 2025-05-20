#include "utils.h"

// Soft-thresholding (shrinkage) operator (shrink(y, lambda * t))
void shrink(double *y, double lambda_t, int n) {
    int sign;
    for (int i = 0; i < n; i++) {
        sign = (y[i] >= 0) ? 1 : -1;
        if (sign * y[i] < lambda_t)
            y[i] = 0;
        else
            y[i] = y[i] - sign * lambda_t;
    }
}

// Matrix-vector multiplication: y = A * x
void mat_vec_mult(double **A, double *x, double *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0;
        for (int j = 0; j < cols; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}
//Compute norm of a vector
double calculate_norm(double *x, int n) {
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += x[i] * x[i];
    }
    return sqrt(norm);
}

// Compute the objective function value: f(x) = ||Ax - b||^2 + lambda_1 * ||x||_1 + lambda_2/2 * ||x||^2
double f(double **A, double *b, double *x, int rows, int cols) {
    double *Ax = (double *)malloc(rows * sizeof(double));
    if (Ax == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    mat_vec_mult(A, x, Ax, rows, cols);

    // Compute ||Ax - b||^2
    double norm = 0;
    for (int i = 0; i < rows; i++) {
        norm += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    }
    free(Ax);

    //compute ||x||_1
    double l1_norm = 0;
    for (int j = 0; j < cols; j++) {
        l1_norm += fabs(x[j]);
    }

    //compute ||x||_2
    double l2_norm = 0;
    for (int j = 0; j < cols; j++) {
        l2_norm += x[j] * x[j];
    }

    return norm + LAMBDA_1 * l1_norm + LAMBDA_2 / 2 * l2_norm;
}

// Compute gradient: grad = A^T * (A*x - b) + lambda_2 * x
double *compute_gradient(double **A, double *b, double *x, double *grad, int rows, int cols) {
    double *tmp_matrix = (double *)malloc(rows * sizeof(double));
    if (tmp_matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL; 
    }
    mat_vec_mult(A, x, tmp_matrix, rows, cols);

    for (int i = 0; i < rows; i++) {
        tmp_matrix[i] -= b[i];  // Compute Ax - b
    }

    for (int j = 0; j < cols; j++) {
        grad[j] = 0;
        // Compute A^T * (Ax - b)
        for (int i = 0; i < rows; i++) {
            grad[j] += A[i][j] * tmp_matrix[i];
        }
        // Add lambda_2 * x
        grad[j] += LAMBDA_2 * x[j]; 
    }

    // Free temporary memory
    free(tmp_matrix);
    return grad;
}

//returns the step size with inexact line search backtracking
double back_tracking_line_search(double **A, double *b, double *x, double *grad, int rows, int cols) {
    double t = T_0; // initial step size

    // Compute the initial function value
    double f_x = f(A, b, x, rows, cols);

    double *x_new = (double *)malloc(cols * sizeof(double));
    if (x_new == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    int iter = 0;
    while (1) {
        // Compute the new x
        for (int i = 0; i < cols; i++) {
            x_new[i] = x[i] - t * grad[i];
        }

        // Compute the new function value
        double f_new = f(A, b, x_new, rows, cols);
        
        // Check the Armijo condition
        double armijo_condition = f_x - f_new;
        double grad_dot = 0;
        // Compute grad(f(x)) * direction
        for (int i = 0; i < cols; i++) {
            grad_dot += grad[i] * (x[i] - x_new[i]);
        }
        if (armijo_condition >= C_ARMIJO * t * grad_dot) {
            break; // Armijo condition satisfied
        }
        
        // if not, reduce the step size
        t *= ALPHA;

        // Free the temporary x_new
        iter++;
    }
    //printf("armijo found in %d iterations\n", iter);
    free(x_new);
    return t;
}