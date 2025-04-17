#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 1000
#define TOL 1e-7

// Soft-thresholding (shrinkage) operator (shrink(y, lambda * t))
void shrink(double *y, double lambda_t, int n) {
    int sign;
    for (int i = 0; i < n; i++) {
        sign = (y[i] > 0) ? 1 : -1;
        if (sign * y[i] < lambda_t)
            y[i] = 0;
        else
            y[i] = y[i] - sign * lambda_t;
    }
}

// Matrix-vector multiplication: y = A * x
void mat_vec_mult(double *A, double *x, double *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0;
        for (int j = 0; j < cols; j++) {
            y[i] += A[i * cols + j] * x[j];
        }
    }
}

// Compute gradient: grad = A^T * (A*x - b)
void compute_gradient(double *A, double *b, double *x, double *grad, int rows, int cols) {
    double *tmp_matrix = (double *)malloc(rows * sizeof(double));
    mat_vec_mult(A, x, tmp_matrix, rows, cols);

    for (int i = 0; i < rows; i++) {
        tmp_matrix[i] -= b[i];  // Compute Ax - b
    }

    // Compute A^T * (Ax - b)
    for (int j = 0; j < cols; j++) {
        grad[j] = 0;
        for (int i = 0; i < rows; i++) {
            grad[j] += A[i * cols + j] * tmp_matrix[i];
        }
    }

    free(tmp_matrix);
}


//leipschitz constant computation
//double check if this is correct
double compute_lipschitz(double *A, int rows, int cols) {
    double max_norm = 0;
    for (int j = 0; j < cols; j++) {
        double col_norm = 0;
        for (int i = 0; i < rows; i++) {
            col_norm += A[i * cols + j] * A[i * cols + j];
        }
        if (col_norm > max_norm) {
            max_norm = col_norm;
        }
    }
    return max_norm;
}

// ISTA Algorithm
void ista(double *A, double *b, double *x, int rows, int cols, double lambda) {

    double t_k = 1.0 / compute_lipschitz(A, rows, cols); // Lipschitz constant
    double *grad = (double *)malloc(cols * sizeof(double));
    double *x_new = (double *)malloc(cols * sizeof(double));

    for (int iter = 0; iter < MAX_ITER; iter++) {
        compute_gradient(A, b, x, grad, rows, cols);

        // x_new = x - t_k * grad g(x_k)
        for (int i = 0; i < cols; i++) {
            x_new[i] = x[i] - t_k * grad[i];
        }
        //shrink (x_k - t_k * grad; lambda * t_k)
        shrink(x_new, lambda * t_k, cols);

        // Check convergence ||x_new - x|| < TOL
        double norm_diff = 0;
        for (int i = 0; i < cols; i++) {
            norm_diff += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        printf("Iteration %d: norm_diff = %f\n", iter, sqrt(norm_diff));
        if (sqrt(norm_diff) < TOL) {
            break;
        }

        // Update x
        for (int i = 0; i < cols; i++) {
            x[i] = x_new[i];
        }
    }

    free(grad);
    free(x_new);
}

// Example Usage
int main() {
    int rows = 100, cols = 20;
    double lambda = 0.1;

    // Generate random A and b
    double *A = (double *)malloc(rows * cols * sizeof(double));
    double *b = (double *)malloc(rows * sizeof(double));
    double *x = (double *)calloc(cols, sizeof(double)); // Initialize x to 0 (initial solution)

    for (int i = 0; i < rows * cols; i++) {
        A[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random values in [-1, 1]
    }
    for (int i = 0; i < rows; i++) {
        b[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }


    // Run ISTA
    ista(A, b, x, rows, cols, lambda);

    // Print result
    printf("Optimized x:\n");
    for (int i = 0; i < cols; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");

    // Free memory
    free(A);
    free(b);
    free(x);

    return 0;
}
