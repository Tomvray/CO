#ifndef PARAMETERS_H
   #define PARAMETERS_H

    // Parameters for LASSO OR Elastic Net
    #define LAMBDA_1 0
    #define LAMBDA_2 0.1
    
    // Parameters for the simulation
    #define MAX_ITER 1e5
    #define TOL 1e-4
    #define T_0 1.0

    // Parameters for armijo condition
    #define C_ARMIJO 1e-4 // Armijo condition constant
    #define ALPHA 0.6 // contraction factor

#endif