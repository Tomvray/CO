#ifndef PARAMETERS_H
   #define PARAMETERS_H

    // Parameters for LASSO OR Elastic Net
    #define LAMBDA_1 0
    #define LAMBDA_2 0.2
    
    // Parameters for the simulation
    #define MAX_ITER 1e3
    #define TOL 1e-4
    #define T_0 1.0

    // Parameters for armijo condition
    #define C_ARMIJO 0.5 // Armijo condition constant
    #define ALPHA 0.4// contraction factor

#endif