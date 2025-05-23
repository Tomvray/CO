#include "algorithms.h"

// ISTA Algorithm
double* ista(double *x, Problem problem, FILE *file) {

    // double t_k = 1.0 / compute_lipschitz(A, rows, cols); // Lipschitz constant
    Data data = problem.data;
    double t_k = data.t_0;
    int cols = data.cols;
    int rows = data.rows;
    double **A = data.A;
    double *b = data.b;
    
    double *grad = (double *)malloc(cols * sizeof(double));
    double *x_new = (double *)malloc(cols * sizeof(double));
    if (grad == NULL || x_new == NULL) {
        printf("Error: Memory allocation failed\n");
        return NULL;
    }
    // Initialize x_new
    for (int i = 0; i < cols; i++) x_new[i] = x[i];
    // Initialize gradient
    for (int i = 0; i < cols; i++) grad[i] = 0.0;

    // Start the timer
    clock_t start_time = clock();
    if (file != NULL) {
        // Write header to the file
        fprintf(file, "Iteration,t_k,gradnorm,objective,time\n");
    }
    
    for (int iter = 0; iter < MAX_ITER; iter++) {

        // compute gradient g(x_k) = A^T(Ax_k - b)
        compute_gradient(A, b, x, grad, rows, cols);

        // compute new t_k
        //t_k = back_tracking_line_search(x, grad, data); 

        // x_new = x - t_k * grad g(x_k)
        for (int i = 0; i < cols; i++) {
            x_new[i] = x[i] - t_k * grad[i];
        }
        //shrink (x_k - t_k * grad; LAMBDA_1 * t_k)
        shrink(x_new, LAMBDA_1 * t_k, cols);

        if (file != NULL) {
            // Write iteration data to the file
            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            fprintf(file, "%d,%f,%f,%f,%f\n", iter, t_k, calculate_norm(grad, cols), f(A, b, x, rows, cols), elapsed_time);
            fflush(file); // Flush the file buffer to ensure data is written immediately
        }

        // Check convergence |f(x_k) - f(x_new)| < TOL
        if (fabs(f(A, b, x, rows, cols) - f(A, b, x_new, rows, cols)) < TOL) {
            printf("ISTA converged at iteration %d\n", iter);
            break;
        }

        // Update x
        for (int i = 0; i < cols; i++) {
            x[i] = x_new[i];
        }
    }

    printf("ISTA converged at iteration with value of f : %f\n", f(A, b, x, rows, cols));
    free(grad);
    free(x_new);

    return x;
}

// FISTA Algorithm
double* fista(double *x, Problem problem, FILE *file) {
    Data data = problem.data;
    double **A = data.A;
    double *b = data.b;
    int rows = data.rows;
    int cols = data.cols;

    double t_k = data.t_0;
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

    // Initialize gradient
    for (int i = 0; i < cols; i++) grad[i] = 0.0;

    // Initialize x_new
    for (int i = 0; i < cols; i++) x_new[i] = x[i];

    // Start the timer
    clock_t start_time = clock();
    if (file != NULL) {
        // Write header to the file
        fprintf(file, "Iteration,t_k,gradnorm,objective,time\n");
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // compute gradient g(y_k) = A^T(Ay_k - b)
        compute_gradient(A, b, y, grad, rows, cols);

        // compute new t_k
        // t_k = back_tracking_line_search(x, grad, data);

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
        
        if (file != NULL) {
            // Write iteration data to the file
            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            fprintf(file, "%d,%f,%f,%f,%f\n", iter, t_k, calculate_norm(grad, cols), f(A, b, x, rows, cols), elapsed_time);
            fflush(file); // Flush the file buffer to ensure data is written immediately
        }

        // Check convergence | f(x_k) - f(x_new)| < TOL
        if (fabs(f(A, b, x, rows, cols) - f(A, b, x_new, rows, cols)) < TOL) {
            printf("FISTA converged at iteration %d\n", iter);
            for (int i = 0; i < cols; i++) x[i] = x_new[i];
            break;
        }
        // Update x
        for (int i = 0; i < cols; i++) x[i] = x_new[i];
        
        // Update momentum factor
        momentum_factor = momentum_factor_new;
    }

    printf("FISTA converged at iteration with value of f :%f\n", f(A, b, x, rows, cols));
    free(grad);
    free(x_new);
    free(y);
    return x;
}

//typedef double (*objective_func)(double **A, double *b, double *x, int rows, int cols);
//Compute the proximal gradiant using L-BFGS

// Proximal operator using BFGS (simplified, not full BFGS implementation)
/*double* prox_bfgs(double *v, double lambda, int n, void (*grad_g)(double*, double*, int), int max_iter, double tol) {
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
    }*/
   
static int printed = 0;
// L-BFGS implementation
double* L_BFGS(double* x, int m, Problem problem, FILE *file) {
    Data data = problem.data;
    int dimensions = data.cols;
    double t[m], rho[m];
    double *q = malloc(dimensions * sizeof(double));
    double *grad_new = malloc(dimensions * sizeof(double));
    double **s, **y;

    s = malloc(m * sizeof(double*));
    y = malloc(m * sizeof(double*));

    for (int i = 0; i < m; i++) {
        s[i] = malloc(dimensions * sizeof(double));
        y[i] = malloc(dimensions * sizeof(double));
    }

    double *grad_old = malloc(dimensions * sizeof(double));
    double *x_old = malloc(dimensions * sizeof(double));
    

    problem.grad_func(x, grad_old, data);
    if (grad_old == NULL) {
        printf("Error: Memory allocation failed\n");
        return NULL;
    }
    int k = 0;

    // Start the timer
    clock_t start_time = clock();
    if (file != NULL) {
        // Write header to the file
        fprintf(file, "Iteration,t_k,gradnorm,objective,time\n");
    }

    while (k < MAX_ITER ) {
        
        // q <- grad(f(x_k))
        memcpy(q, grad_old, dimensions * sizeof(double));

        int upper_bound = (k < m) ? k : m;
        // t_i <- rho_i * s_i^T q
        // q <- q - t_i * y_i
        for (int i = upper_bound - 1; i >= 0; i--) {
            rho[i] = 1.0 / dot_product(y[i], s[i], dimensions);
            t[i] = rho[i] * dot_product(s[i], q, dimensions);
            for (int j = 0; j < dimensions; j++)
                q[j] -= t[i] * y[i][j];
        }

        double gamma = 1.0;
        // lambda <- s_{m-1}^T y_{m-1} / y_{m-1}^T y_{m-1}
        // B_k^0 = gamma * I
        if (k > 0) {
            gamma = dot_product(s[upper_bound - 1], y[upper_bound - 1], dimensions) /
                    dot_product(y[upper_bound - 1], y[upper_bound - 1], dimensions);
        }


        // q <- gamma * q (where q stores r)
        for (int i = 0; i < dimensions; i++)
            q[i] *= gamma;

        // b <- rho_i * y_i^T q
        // q <- q + s_i * (t_i - b)
        for (int i = 0; i < upper_bound; i++) {
            double b = rho[i] * dot_product(y[i], q, dimensions);
            for (int j = 0; j < dimensions; j++)
                q[j] += s[i][j] * (t[i] - b);
        }


        for (int i = 0; i < dimensions; i++) {
            x_old[i] = x[i];
            x[i] -= q[i];
        }


        problem.grad_func(x, grad_new, data);
        if (grad_new == NULL) {
            printf("Error: Memory allocation failed\n");
            return NULL;
        }


        if (k >= m) {
            free(s[0]);
            free(y[0]);
            memmove(s, s+1, (m-1)*sizeof(double*));
            memmove(y, y+1, (m-1)*sizeof(double*));
            s[m-1] = malloc(dimensions * sizeof(double));
            y[m-1] = malloc(dimensions * sizeof(double));
            for (int i = 0; i < dimensions; i++) {
                s[m-1][i] = x[i] - x_old[i];
                y[m-1][i] = grad_new[i] - grad_old[i];
            }
        }
        else{
            for (int i = 0; i < dimensions; i++) {
                s[k][i] = x[i] - x_old[i];
                y[k][i] = grad_new[i] - grad_old[i];
            }
        }


        memcpy(grad_old, grad_new, dimensions * sizeof(double));

        k++;

        if (file != NULL) {
            // Write iteration data to the file
            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            fprintf(file, "%d,%f,%f,%f,%f\n", k, 0.0, calculate_norm(grad_old, dimensions), problem.objective_func(x, data), elapsed_time);
            fflush(file); // Flush the file buffer to ensure data is written immediately
        }

        // Check convergence |f(x_k) - f(x_new)| < TOL
        if (fabs(problem.objective_func(x, data) - problem.objective_func(x_old, data)) < TOL) {
            if (printed == 0) {
                printf("L-BFGS converged at iteration %i with value of f :%f\n", k, problem.objective_func(x, data));
                printed = 1;
            }
            break;
        }
    }

    free(q);
    free(grad_new);
    free(grad_old);
    free(x_old);
    for (int i = 0; i < m; i++) {
        free(s[i]);
        free(y[i]);
    }
    free(s);
    free(y);

    if (printed == 0) {
        printf("L-BFGS converged at iteration %i with value of f :%f\n", k, problem.objective_func(x, data));
        printed = 1;
    }
    return x;
}

double* grad_LS(double *x, double *grad, Data data) {
    double **A = data.A;
    double *b = data.b;
    int rows = data.rows;
    int cols = data.cols;

    // Compute gradient: grad = A^T * (A*x - b)
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
    }

    // Free temporary memory
    free(tmp_matrix);
    return grad;
}


double* LBFGS_fista(double *x, int m, Problem problem, FILE *file) {
    Data data = problem.data;
    double **A = data.A;
    double *b = data.b;
    int rows = data.rows;
    int cols = data.cols;

    double t_k = data.t_0;
    double momentum_factor = 1.0;
    double momentum_factor_new;

    double *grad = (double *)malloc(cols * sizeof(double));
    double *x_new = (double *)malloc(cols * sizeof(double));
    double *y = (double *)malloc(cols * sizeof(double));
    double *u = (double*) malloc(cols * sizeof(double));
    if (grad == NULL || x_new == NULL || y == NULL || u == NULL) {
        printf("Error: Memory allocation failed\n");
        return NULL;
    }
    // Initialize y
    for (int i = 0; i < cols; i++) y[i] = x[i];

    // Initialize gradient
    for (int i = 0; i < cols; i++) grad[i] = 0.0;

    // Initialize x_new
    for (int i = 0; i < cols; i++) x_new[i] = x[i];

    // Start the timer
    clock_t start_time = clock();
    if (file != NULL) {
        // Write header to the file
        fprintf(file, "Iteration,t_k,gradnorm,objective,time\n");
    }
    
    for (int iter = 0; iter < MAX_ITER; iter++) {

        // compute gradient g(y_k) = A^T(Ay_k - b)
        grad_LS(y, grad, data);

        // compute new t_k
        //t_k = back_tracking_line_search(x, grad, data);

        // x_new = y - t_k * grad g(y_k)
        for (int i = 0; i < cols; i++){
            u[i] = y[i] - t_k * grad[i];
            //x_new[i] = u[i];
        }
        
        //compute prox
        Problem sub_prox_problem;
        sub_prox_problem.data = problem.data;
        sub_prox_problem.data.x = u;
        sub_prox_problem.objective_func = prox_function;
        sub_prox_problem.grad_func = prox_gradient;

        L_BFGS(x_new, m, sub_prox_problem, NULL);

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
        
        if (file != NULL) {
            // Write iteration data to the file
            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            fprintf(file, "%d,%f,%f,%f,%f\n", iter, t_k, calculate_norm(grad, cols), f(A, b, x, rows, cols), elapsed_time);
            fflush(file); // Flush the file buffer to ensure data is written immediately
        }

        // Check convergence | f(x_k) - f(x_new)| < TOL
        if (fabs(f(A, b, x, rows, cols) - f(A, b, x_new, rows, cols)) < TOL) {
            printf("LBFGS FISTA converged at iteration %d\n", iter);
            break;
        }
        // Update x
        for (int i = 0; i < cols; i++) x[i] = x_new[i];
        
        // Update momentum factor
        momentum_factor = momentum_factor_new;
    }

    printf("LBFGS FISTA converged at iteration with value of f :%f\n", f(A, b, x, rows, cols));
    free(grad);
    free(x_new);
    free(y);
    free(u);
    return x;
}


//DUAL FISTA

// Compute the gradient of f^*(-A^T y) = -A(A^T y + b)
void grad_f_star(double* grad, double* A, double* ATy, double* b, int m, int n) {
    for (int i = 0; i < m; i++) {
        double temp = 0;
        for (int j = 0; j < n; j++) {
            temp += A[i * n + j] * (ATy[j] + b[j]);
        }
        grad[i] = -temp;
    }
}

// Project y onto l_infinity ball of radius lambda
void proj_linf_ball(double* y_proj, double* y, int m, double lambda) {
    for (int i = 0; i < m; i++) {
        if (y[i] > lambda)
            y_proj[i] = lambda;
        else if (y[i] < -lambda)
            y_proj[i] = -lambda;
        else
            y_proj[i] = y[i];
    }
}

// Dual FISTA
void dual_fista(double* y, double* A, double* b, int m, int n, double lambda, int max_iter) {
    double* y_old = (double*)calloc(m, sizeof(double));
    double* z = (double*)calloc(m, sizeof(double));
    double* grad = (double*)calloc(m, sizeof(double));
    double* ATy_buf = (double*)calloc(n, sizeof(double));
    double t = 1, t_old;
    double t_k;

    for (int k = 0; k < max_iter; k++) {
        // Compute A^T z
        ATy(ATy_buf, A, z, m, n);

        // Compute gradient
        grad_f_star(grad, A, ATy_buf, b, m, n);

        t_k = 1.0 / (k + 1); // Step size

        // Gradient step
        for (int i = 0; i < m; i++) {
            y[i] = z[i] - grad[i] / t_k;
        }

        // Projection
        proj_linf_ball(y, y, m, lambda);

        // Update momentum
        t_old = t;
        t = (1 + sqrt(1 + 4 * t_old * t_old)) / 2;
        for (int i = 0; i < m; i++) {
            z[i] = y[i] + ((t_old - 1) / t) * (y[i] - y_old[i]);
            y_old[i] = y[i];
        }
    }

    free(y_old);
    free(z);
    free(grad);
    free(ATy_buf);
}
