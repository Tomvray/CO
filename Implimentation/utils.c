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

// Compute data.prox_function(u) + 1/2 * ||u - Problem.x||^2
// compute h(u) + 1/2 * ||u - x||^2
double prox_function(double *u, Data data){
    double norm = 0;
    for (int i = 0; i < data.cols; i++) {
        norm += (u[i] - data.x[i]) * (u[i] - data.x[i]);
    }

    return data.prox_func(u, data) + 0.5 * norm;
}

// Compute the gradient of the proximal function data.grad_prox_function(u) + (u - data.x)
double* prox_gradient(double *u, double* grad, Data data){
    double* grad_function_to_prox = (double *)malloc(data.cols * sizeof(double)); // grad(h(u)) 
    if (grad_function_to_prox == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    data.prox_grad(u, grad_function_to_prox, data);
    for (int i = 0; i < data.cols; i++) {
        grad[i] = grad_function_to_prox[i] + (u[i] - data.x[i]);
    }

    free(grad_function_to_prox);
    return grad;
}

// l2
double l2_func (double* x, Data data){
    double norm = 0;
    for (int i = 0; i < data.cols; i++) {
        norm += x[i] * x[i];
    }
    return norm * LAMBDA_2 * 0.5;
}

// gradient of l2
double* l2_grad (double* x, double* grad, Data data){
    for (int i = 0; i < data.cols; i++) {
        grad[i] = LAMBDA_2 * x[i];
    }
    return grad;
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
    // return norm;
    return sqrt(norm);
}

// Dot product function
double dot_product(double* v1, double* v2, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }
    return result;
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
        if (x[j] < 0)
            l1_norm += -x[j];
        else
            l1_norm += x[j];
    }

    //compute ||x||_2
    double l2_norm = 0;
    for (int j = 0; j < cols; j++) {
        l2_norm += x[j] * x[j];
    }
    // printf("square %f, l1_norm: %f, l2_norm: %f\n", norm, l1_norm, l2_norm);

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
double back_tracking_line_search(double *x, double *grad, Data data) {
    
    double **A = data.A;
    double *b = data.b;
    int rows = data.rows;
    int cols = data.cols;
    double t = data.t_0;    

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
        
        shrink(x_new, LAMBDA_1 * t, cols);
        
        // printf("iter: %i, t: %f\n", iter, t);
        // Compute the new function value
        double f_new = f(A, b, x_new, rows, cols);
        
        
        // Check the Armijo condition
        double armijo_condition = f_x - f_new;
        /*double grad_dot = 0;
        // Compute grad(f(x)) * direction
        for (int i = 0; i < cols; i++) {
            grad_dot += grad[i] * (x[i] - x_new[i]);
            }*/
            
            
        if (armijo_condition > C_ARMIJO * t * dot_product(grad, grad, cols)) break; // Armijo condition satisfied
        // if not, reduce the step size
        t  = t * ALPHA;
        // if (iter < 1)
            // printf("t_k: %f, f_new: %f, iter: %i, armijo_condition: %f, grad_dot: %f\n", t, f_new, iter, armijo_condition, grad);
        // Free the temporary x_new
        iter++;
    }
    //printf("armijo found in %d iterations\n", iter);
    free(x_new);
    return t;
}

/* --- Power-iteration estimate of ‖AᵀA‖₂ ---------------------------------- */
double power_iteration(double **A, int rows, int cols,
                              int it_max, double tol)
{
    double *v  = malloc(cols * sizeof *v);
    double *Av = malloc(rows * sizeof *Av);
    double *w  = malloc(cols * sizeof *w);
    if (!v || !Av || !w) { perror("malloc"); exit(EXIT_FAILURE); }

    /* start with ‖v‖₂ = 1 */
    for (int j = 0; j < cols; ++j) v[j] = 1.0 / sqrt(cols);

    double lambda_old = 0.0;
    for (int it = 0; it < it_max; ++it) {

        /* w = Aᵀ(A v) */
        mat_vec_mult(A, v, Av, rows, cols);    /* Av = A v           */
        for (int j = 0; j < cols; ++j) {
            w[j] = 0.0;
            for (int i = 0; i < rows; ++i) w[j] += A[i][j] * Av[i];
        }

        /* normalise w → v */
        double norm_w = sqrt(dot_product(w, w, cols));
        for (int j = 0; j < cols; ++j) v[j] = w[j] / norm_w;

        /* Rayleigh quotient gives current eigen-value estimate */
        double lambda = norm_w;
        if (fabs(lambda - lambda_old) / lambda < tol) break;
        lambda_old = lambda;
    }

    free(v); free(Av); free(w);
    return lambda_old;           /* ≈ largest eigen-value of AᵀA */
}

/* Public helper: initial step size ---------------------------------------- */
double estimate_t0(double **A, int rows, int cols)
{
    /* 100 power-iterations are plenty for ≤ 1 e-3 relative accuracy */
    const int    it_max = 100;
    const double tol    = 1e-3;

    double L = power_iteration(A, rows, cols, it_max, tol);
    if (L <= 0.0) {
        fprintf(stderr, "estimate_t0: non-positive Lipschitz constant\n");
        exit(EXIT_FAILURE);
    }
    return 1.0 / L;
}