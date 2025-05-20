#include "algorithms.h"

void create_data(double **A, double *b, int rows, int cols) {
    // Create data for the ISTA and FISTA algorithms

    //fill data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) A[i][j] = (double)rand() / (double)RAND_MAX;   /* uniform in (0,1) */
        // double perturbation = (double)rand() / (double)RAND_MAX; // Random perturbation
        b[i] = 0.2 * A[i][2] + 0.3 * A[i][4]; //+ perturbation; // Linear combination with noise
        
    }
}
int main() {

    // Example usage of the ISTA and FISTA algorithms
    int rows = 10; // Number of rows in A
    int cols = 10;  // Number of columns in A
    double **A = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) A[i] = (double *)malloc(cols * sizeof(double));
    double *b = (double *)malloc(rows * sizeof(double));
    double *x_ista = (double *)malloc(cols * sizeof(double));
    double *x_fista = (double *)malloc(cols * sizeof(double));
    if (A == NULL || b == NULL || x_ista == NULL || x_fista == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    // Initialize A, b, and x with random values
    create_data(A, b, rows, cols);
    
    // Initialize x with zeros
    for (int i = 0; i < cols; i++) x_ista[i] = 0.0;

    // Initialize x with zeros
    for (int i = 0; i < cols; i++) x_fista[i] = 0.0;
    
    // Run ISTA
    x_ista = ista(x_ista, A, b, rows, cols);
    
    // Run FISTA
    x_fista = fista(x_fista, A, b, rows, cols);

    // Print results
    printf("ista\n");
    //print vectors
    for (int i = 0; i < cols; i++) {
        printf("%f,", x_ista[i]);
    }
    printf("\n");

    printf("\nFista\n");
    //print vectors
    for (int i = 0; i < cols; i++) {
        printf("%f, ", x_fista[i]);
    }
    printf("\n");
    
    // Free allocated memory
    for (int i = 0; i < rows; i++) free(A[i]);
    free(A);
    free(b);
    free(x_ista);
    free(x_fista);
    return 0;
}