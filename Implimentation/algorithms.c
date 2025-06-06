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
        fprintf(file, "Iteration,t_k,gradnorm,objective,time,grad_time,prox_time\n");
    }
    
    for (int iter = 0; iter < MAX_ITER; iter++) {

        clock_t grad_start_time = clock();
        // compute gradient g(x_k) = A^T(Ax_k - b)
        compute_gradient(A, b, x, grad, rows, cols);
        clock_t grad_end_time = clock();
        double grad_time = (double)(grad_end_time - grad_start_time);

        // compute new t_k
        //t_k = back_tracking_line_search(x, grad, data); 

        // x_new = x - t_k * grad g(x_k)
        for (int i = 0; i < cols; i++) {
            x_new[i] = x[i] - t_k * grad[i];
        }
        clock_t prox_start_time = clock();
        //shrink (x_k - t_k * grad; LAMBDA_1 * t_k)
        shrink(x_new, LAMBDA_1 * t_k, cols);
        clock_t prox_end_time = clock();
        double prox_time = (double)(prox_end_time - prox_start_time);

        if (file != NULL) {
            // Write iteration data to the file
            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time);
            fprintf(file, "%d,%f,%f,%f,%f,%f,%f\n", iter, t_k, calculate_norm(grad, cols), f(A, b, x, rows, cols), elapsed_time, grad_time, prox_time);
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
        fprintf(file, "Iteration,t_k,gradnorm,objective,time,grad_time,prox_time\n");
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {

        clock_t grad_start_time = clock();
        // compute gradient g(y_k) = A^T(Ay_k - b)
        compute_gradient(A, b, y, grad, rows, cols);
        clock_t grad_end_time = clock();
        double grad_time = (double)(grad_end_time - grad_start_time);

        // compute new t_k
        //t_k = back_tracking_line_search(x, grad, data);
        
        // x_new = y - t_k * grad g(y_k)
        for (int i = 0; i < cols; i++)
            x_new[i] = y[i] - t_k * grad[i];
        
        clock_t prox_start_time = clock();
        shrink(x_new, LAMBDA_1 * t_k, cols);
        clock_t prox_end_time = clock();
        double prox_time = (double)(prox_end_time - prox_start_time);

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
            double elapsed_time = (double)(current_time - start_time);
            fprintf(file, "%d,%f,%f,%f,%f,%f,%f\n", iter, t_k, calculate_norm(grad, cols), f(A, b, x, rows, cols), elapsed_time, grad_time, prox_time);
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
   
static int printed = 0;
// L-BFGS implementation
double* L_BFGS(double* x, int m, Problem problem, FILE *file) {
    Data data = problem.data;
    int dimensions = data.cols;
    double t[m], rho[m];
    double *q = malloc(dimensions * sizeof(double));
    double *grad_new = malloc(dimensions * sizeof(double));
    double **s, **y;
    double t_k = 0;

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
        fprintf(file, "Iteration,t_k,gradnorm,objective,time,two_loop_time,grad_time\n");
    }

    while (k < MAX_ITER ) {

        
        // q <- grad(f(x_k))
        memcpy(q, grad_old, dimensions * sizeof(double));

        int upper_bound = (k < m) ? k : m;
        // t_i <- rho_i * s_i^T q
        // q <- q - t_i * y_i
        clock_t two_loop_start_time = clock();
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
        clock_t two_loop_end_time = clock();
        double two_loop_time = (double)(two_loop_end_time - two_loop_start_time);

        for (int i = 0; i < dimensions; i++) {
            x_old[i] = x[i];
            x[i] -= q[i];
        }


        // compute new gradient
        clock_t grad_start_time = clock();
        problem.grad_func(x, grad_new, data);
        clock_t grad_end_time = clock();
        double grad_time = (double)(grad_end_time - grad_start_time);
        if (grad_new == NULL) {
            printf("Error: Memory allocation failed\n");
            return NULL;
        }

        // update m matrix
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
            double elapsed_time = (double)(current_time - start_time);
            fprintf(file, "%d,%f,%f,%f,%f,%f,%f\n", k, t_k, calculate_norm(grad_new, dimensions), problem.objective_func(x, data), elapsed_time, two_loop_time, grad_time);
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
        fprintf(file, "Iteration,t_k,gradnorm,objective,time,grad_time,prox_time\n");
    }
    
    for (int iter = 0; iter < MAX_ITER; iter++) {

        // compute gradient g(y_k) = A^T(Ay_k - b)
        clock_t grad_start_time = clock();
        grad_LS(y, grad, data);
        clock_t grad_end_time = clock();
        double grad_time = (double)(grad_end_time - grad_start_time);

        // compute new t_k
        //t_k = back_tracking_line_search(x, grad, data);

        // x_new = y - t_k * grad g(y_k)
        for (int i = 0; i < cols; i++) u[i] = y[i] - t_k * grad[i];
    
        
        //compute prox
        Problem sub_prox_problem;
        sub_prox_problem.data = problem.data;
        sub_prox_problem.data.x = u;
        sub_prox_problem.objective_func = prox_function;
        sub_prox_problem.grad_func = prox_gradient;

        // Call L-BFGS to solve the subproblem
        clock_t prox_start_time = clock();
        L_BFGS(x_new, m, sub_prox_problem, NULL);
        clock_t prox_end_time = clock();
        double prox_time = (double)(prox_end_time - prox_start_time);

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
            double elapsed_time = (double)(current_time - start_time);
            fprintf(file, "%d,%f,%f,%f,%f,%f,%f\n", iter, t_k, calculate_norm(grad, cols), f(A, b, x, rows, cols), elapsed_time, grad_time, prox_time);
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