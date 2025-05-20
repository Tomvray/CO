#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 1000
#define TOL      1e-7

/* ---------- helper routines (unchanged) ---------- */

/* Soft-threshold operator: shrink(y, λ·τ) */
void shrink(double *y, double lambda_t, int n) {
    for (int i = 0; i < n; ++i) {
        double sign = (y[i] >= 0.0) ? 1.0 : -1.0;
        double val  = fabs(y[i]) - lambda_t;
        y[i] = (val > 0.0) ? sign * val : 0.0;
    }
}

/* y = A * x   (row-major storage) */
void mat_vec_mult(const double *A, const double *x,
                  double *y, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j)
            sum += A[i * cols + j] * x[j];
        y[i] = sum;
    }
}

/* grad = Aᵀ (A x − b) */
void compute_gradient(const double *A, const double *b,
                      const double *x, double *grad,
                      int rows, int cols)
{
    double *tmp = (double *)malloc(rows * sizeof(double));

    mat_vec_mult(A, x, tmp, rows, cols);
    for (int i = 0; i < rows; ++i)         /* tmp ← A x − b */
        tmp[i] -= b[i];

    for (int j = 0; j < cols; ++j) {       /* grad ← Aᵀ tmp */
        double sum = 0.0;
        for (int i = 0; i < rows; ++i)
            sum += A[i * cols + j] * tmp[i];
        grad[j] = sum;
    }
    free(tmp);
}

/* crude upper bound on ‖AᵀA‖₂: max column squared-norm */
double compute_lipschitz(const double *A, int rows, int cols)
{
    double max_norm = 0.0;
    for (int j = 0; j < cols; ++j) {
        double col_norm = 0.0;
        for (int i = 0; i < rows; ++i) {
            double a = A[i * cols + j];
            col_norm += a * a;
        }
        if (col_norm > max_norm) max_norm = col_norm;
    }
    return max_norm;          /* L = max col ‖a_j‖²         */
}

/* ---------- FISTA solver ---------- */
void fista(const double *A, const double *b,
           double *x, int rows, int cols, double lambda)
{
    double L     = compute_lipschitz(A, rows, cols);
    double tau   = 1.0 / L;               /* step size      */

    double *y    = (double *)malloc(cols * sizeof(double));
    double *x_new= (double *)malloc(cols * sizeof(double));
    double *grad = (double *)malloc(cols * sizeof(double));

    for (int i = 0; i < cols; ++i) y[i] = x[i];    /* y₀ = x₀   */
    double t_old = 1.0;

    for (int k = 0; k < MAX_ITER; ++k) {

        /* 1. gradient at momentum point yᵏ */
        compute_gradient(A, b, y, grad, rows, cols);

        /* 2. proximal step */
        for (int i = 0; i < cols; ++i)
            x_new[i] = y[i] - tau * grad[i];

        shrink(x_new, lambda * tau, cols);        /* prox_{λτ}  */

        /* 3. convergence test ‖x_new − x‖ */
        double diff = 0.0;
        for (int i = 0; i < cols; ++i) {
            double d = x_new[i] - x[i];
            diff += d * d;
        }
        diff = sqrt(diff);
        printf("Iter %4d  ‖x_{k+1}−x_k‖ = %.3e\n", k, diff);
        if (diff < TOL) break;

        /* 4. Nesterov momentum update */
        double t_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t_old * t_old));
        double beta  = (t_old - 1.0) / t_new;
        for (int i = 0; i < cols; ++i)
            y[i] = x_new[i] + beta * (x_new[i] - x[i]);

        /* 5. shift variables for next iteration */
        for (int i = 0; i < cols; ++i) x[i] = x_new[i];
        t_old = t_new;
    }

    free(y); free(x_new); free(grad);
}

/* ---------- demo main ---------- */
int main(void)
{
    const int rows = 100, cols = 20;
    const double lambda = 0.1;

    double *A = (double *)malloc(rows * cols * sizeof(double));
    double *b = (double *)malloc(rows * sizeof(double));
    double *x = (double *)calloc(cols, sizeof(double));  /* x₀ = 0 */

    /* random test data */
    for (int i = 0; i < rows * cols; ++i)
        A[i] = 2.0 * rand() / RAND_MAX - 1.0;
    for (int i = 0; i < rows; ++i)
        b[i] = 2.0 * rand() / RAND_MAX - 1.0;

    fista(A, b, x, rows, cols, lambda);

    puts("\nOptimised x:");
    for (int j = 0; j < cols; ++j)
        printf("%.6f ", x[j]);
    putchar('\n');

    free(A); free(b); free(x);
    return 0;
}