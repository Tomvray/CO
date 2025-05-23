#ifndef PARAMETERS_H
   #define PARAMETERS_H

    // Parameters for LASSO OR Elastic Net
    #define LAMBDA_1 0
    #define LAMBDA_2 0
    
    
    // Parameters for the simulation
    #define MAX_ITER 1e5
    #define TOL 1e-8
    #define T_0 1.0

    // Parameters for armijo condition
    #define C_ARMIJO 0.7 // Armijo condition constant
    #define ALPHA 0.6 // contraction factor

#endif