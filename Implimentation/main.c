#include "algorithms.h"


void create_data(double **A, double *b, int rows, int cols) {
    // Create data for the ISTA and FISTA algorithms

    //fill data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) A[i][j] = (double)rand() / (double)RAND_MAX;   /* uniform in (0,1) */
        // double perturbation = (double)rand() / (double)RAND_MAX; // Random perturbation
        b[i] = 2 * A[i][2] + 3 * A[i][4]; //+ perturbation; // Linear combination with noise
    }
}

double l2_obj(double *x, double **A, double *b, int rows, int cols) {
    double obj = 0.0;
    double l2_norm = 0.0;

    // Compute ||Ax - b||^2
    double *Ax = (double *)malloc(rows * sizeof(double));
    mat_vec_mult(A, x, Ax, rows, cols);
    for (int i = 0; i < rows; i++) {
        obj += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    }

    // Compute ||x||_1 and ||x||_2
    for (int j = 0; j < cols; j++) {
        l2_norm += x[j] * x[j];
    }

    free(Ax);
    return obj + LAMBDA_2 / 2 * l2_norm;
}


/*double *l2_grad(double *x, double **A, double *b, int rows, int cols) {
    double *grad = (double *)malloc(cols * sizeof(double));
    if (grad == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    // Compute gradient: grad = A^T * (A*x - b) + lambda_2 * x
    compute_gradient(A, b, x, grad, rows, cols);
    return grad;
}*/

double* grad(double *x, double* grad, Data data) {
    return compute_gradient(data.A, data.b, x, grad, data.rows, data.cols);
}

double obj(double *x, Data data) {
    return f(data.A, data.b, x, data.rows, data.cols);
}


// Example gradient of the proximal objective: grad = x - v + lambda * grad_g(x)
void prox_objective_grad(double *x, double *v, double lambda, int n, void (*grad_g)(double*, double*, int), double *grad) {
    double *grad_gx = (double*)malloc(n * sizeof(double));
    grad_g(x, grad_gx, n); // User-supplied gradient of g at x
    for (int i = 0; i < n; i++) {
        grad[i] = x[i] - v[i] + lambda * grad_gx[i];
    }
    free(grad_gx);
}


int main() {

    // Example usage of the ISTA and FISTA algorithms
    int rows = 100; // Number of rows in A
    int cols = 100;  // Number of columns in A
    double **A = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) A[i] = (double *)malloc(cols * sizeof(double));
    double *b = (double *)malloc(rows * sizeof(double));
    double *x_ista = (double *)malloc(cols * sizeof(double));
    double *x_fista = (double *)malloc(cols * sizeof(double));
    double *x_lbfgs = (double *)malloc(cols * sizeof(double));
    double *x_lbfg_fista = (double *)malloc(cols * sizeof(double));
    if (A == NULL || b == NULL || x_ista == NULL || x_fista == NULL || x_lbfgs == NULL || x_lbfg_fista == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    // Initialize A, b, and x with random values
    create_data(A, b, rows, cols);

    Data data;
    data.A = A;
    data.b = b;
    data.rows = rows;
    data.cols = cols;
    data.t_0 = estimate_t0(A, rows, cols);
    data.prox_func = l2_func;
    data.prox_grad = l2_grad;

    Problem problem;
    problem.data = data;
    problem.objective_func = obj;
    problem.grad_func = grad;
    
    printf("t_0: %f\n", data.t_0);
    // Initialize x with zeros
    for (int i = 0; i < cols; i++) x_ista[i] = 1.0;

    // Initialize x with zeros
    for (int i = 0; i < cols; i++) x_fista[i] = 1.0;

    // Initialize x with zeros
    for (int i = 0; i < cols; i++) x_lbfgs[i] = 1.0;

    // Initialize x with zeros
    for (int i = 0; i < cols; i++) x_lbfg_fista[i] = 30.0;

    
    // Run ISTA
    x_ista = ista(x_ista, problem);
    
    // Run FISTA
    x_fista = fista(x_fista, problem);

    // Run LBFGS
    x_lbfgs = L_BFGS(x_lbfgs, 5, problem);

    // Run LBFGS with FISTA
    x_lbfg_fista = LBFGS_fista(x_lbfg_fista, 5, problem);
    

    // Print results
    //printf("ista\n");
    //print vectors
    //for (int i = 0; i < cols; i++) {
    //    printf("%f,", x_ista[i]);
    //}
    //printf("\n");

    //printf("\nFista\n");
    //print vectors
    //for (int i = 0; i < cols; i++) {
    //    printf("%f, ", x_fista[i]);
   // }
    //printf("\n");

    //printf("\nLBFGS\n");
    //print vectors
    //for (int i = 0; i < cols; i++) {
    //    printf("%f, ", x_lbfgs[i]);
   // }
    //printf("\n");

    //printf("\nLBFGS with Fista\n");
    //print vectors
    //for (int i = 0; i < cols; i++) {
    //    printf("%f, ", x_lbfg_fista[i]);
   // }
    //printf("\n");

    // Free allocated memory
    for (int i = 0; i < rows; i++) free(A[i]);
    free(A);
    free(b);
    free(x_ista);
    free(x_fista);
    free(x_lbfgs);
    return 0;
}