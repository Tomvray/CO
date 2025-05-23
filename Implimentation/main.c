#include "algorithms.h"

int len(char *str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

void create_data(double **A, double *b, int rows, int cols) {
    // Create data for the ISTA and FISTA algorithms

    //fill data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) A[i][j] = (double)rand() / (double)RAND_MAX;   /* uniform in (0,1) */
        // double perturbation = (double)rand() / (double)RAND_MAX; // Random perturbation
        b[i] = 2 * A[i][2] + 3 * A[i][4]; //+ perturbation; // Linear combination with noise
    }
}

/* -------------------------------------------------------------------------
   Utility: read a purely‑numeric CSV file where the LAST column is the target
   vector b and the remaining columns form the matrix A (dense, row‑major).
   On success returns 0 and sets *A_out, *b_out, *rows_out, *cols_out.
   Memory layout: A is allocated as a contiguous block so it can be freed via
   free(A[0]); free(A);
   ------------------------------------------------------------------------- */
static int read_csv_numeric(const char *path,
                            double ***A_out,
                            double **b_out,
                            int *rows_out,
                            int *cols_out)
{

    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    char line[1<<15];               /* 32 KiB per line  */
    int rows = 0, cols = 0;

    /* Pass 1: count rows & columns */
    if (fgets(line, sizeof line, fp)) {
        /* handle possible UTF‑8 BOM */
        char *p = line;
        if ((unsigned char)p[0]==0xEF && (unsigned char)p[1]==0xBB && (unsigned char)p[2]==0xBF)
            p += 3;
        for (; *p; ++p) if (*p == ',') ++cols;
        ++cols;     /* commas + 1 = columns */
        ++rows;
    }
    while (fgets(line, sizeof line, fp)) ++rows;
    rewind(fp);

    if (cols < 2) {
        fprintf(stderr, "CSV must contain at least 2 columns (features + target)\n");
        fclose(fp);
        return -1;
    }

    int feat_cols = cols - 1;       /* last column = b */

    /* allocate contiguous matrix A and vector b */
    double *block = malloc((size_t)rows * feat_cols * sizeof *block);
    double **A    = malloc((size_t)rows * sizeof *A);
    double *b     = malloc((size_t)rows * sizeof *b);
    if (!block || !A || !b) {
        perror("malloc");
        fclose(fp);
        return -1;
    }
    for (int i = 0; i < rows; ++i) A[i] = block + (size_t)i * feat_cols;

    /* Pass 2: parse numbers */
    int r = 0;
    while (fgets(line, sizeof line, fp) && r < rows) {
        int c = 0;
        char *tok = strtok(line, ",\n\r");
        while (tok && c < cols) {
            double val = strtod(tok, NULL);
            if (c < feat_cols)
                A[r][c] = val;
            else
                b[r] = val;  /* last column */
            ++c;
            tok = strtok(NULL, ",\n\r");
        }
        if (c != cols) {
            fprintf(stderr, "Line %d has %d columns instead of %d – aborting.\n", r+1, c, cols);
            fclose(fp);
            free(block); free(A); free(b);
            return -1;
        }
        ++r;
    }
    fclose(fp);

    *A_out   = A;
    *b_out   = b;
    *rows_out = rows;
    *cols_out = feat_cols;
    return 0;
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


/* ------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));

    /* Default synthetic dimensions */
    int rows = 100;
    int cols = 100;
    double **A = NULL;
    double *b  = NULL;

    /* ----------------------------------------------------------
       If the user supplied a file path, try to load it.
       ---------------------------------------------------------- */
    if (argc > 1) {
        if (read_csv_numeric(argv[1], &A, &b, &rows, &cols) == 0) {
            printf("Loaded %d×%d matrix from ‘%s’ (last column used as target)\n",
                   rows, cols, argv[1]);
        } else {
            fprintf(stderr, "Falling back to random data.\n");
        }
    }

    /* ----------------------------------------------------------
       Fallback: create synthetic data if no CSV was given or
       loading failed.
       ---------------------------------------------------------- */
    if (!A) {
        /* allocate dense matrix */
        A = malloc((size_t)rows * sizeof *A);
        if (!A) { perror("malloc"); return -1; }
        for (int i = 0; i < rows; ++i) {
            A[i] = malloc((size_t)cols * sizeof **A);
            if (!A[i]) { perror("malloc"); return -1; }
        }
        b = malloc((size_t)rows * sizeof *b);
        if (!b) { perror("malloc"); return -1; }

        create_data(A, b, rows, cols);
        printf("Generated synthetic %d×%d matrix.\n", rows, cols);
    }

    /* ---------------------------------------
       Solution vectors (initialised to 0/1)
       --------------------------------------- */
    double *x_ista       = calloc((size_t)cols, sizeof *x_ista);
    double *x_fista      = calloc((size_t)cols, sizeof *x_fista);
    double *x_lbfgs      = calloc((size_t)cols, sizeof *x_lbfgs);
    double *x_lbfg_fista = calloc((size_t)cols, sizeof *x_lbfg_fista);
    if (!x_ista || !x_fista || !x_lbfgs || !x_lbfg_fista) {
        fprintf(stderr, "Memory allocation failed for solution vectors\n");
        return -1;
    }

    for (int j = 0; j < cols; ++j) {
        x_ista[j] = 1.0;
        x_fista[j] = 1.0;
        x_lbfgs[j] = 1.0;
        x_lbfg_fista[j] = 30.0;
    }

    /* ---------------------------------------
       Package data into the structures used by
       the optimisation routines.
       --------------------------------------- */
    Data data = {
        .A = A,
        .b = b,
        .rows = rows,
        .cols = cols,
        .t_0 = estimate_t0(A, rows, cols),
        .prox_func = l2_func,
        .prox_grad = l2_grad
    };

    Problem problem = {
        .data = data,
        .objective_func = obj,
        .grad_func = grad
    };

    printf("t_0: %f\n", data.t_0);

    /* --------------------------- Run algorithms --------------------------- */
    char* word;
    int len_word = 0;
    if (argv[1] != NULL) {
        int lenarg  = len(argv[1]);
        len_word = 0;

        int i = lenarg - 1;
        while (argv[1][i] != '.') i--;
        i--;
        while (argv[1][i] != '/' && i >= 0) {
            len_word++;
            i--;
        }

        word = malloc(len_word * sizeof(char));
        if (!word || len_word <= 0) {
            fprintf(stderr, "Memory allocation failed for word\n");
            return -1;
        }
        if (argv[1][i] == '/') i++;

        printf("i: %d, len_word: %d\n", i, len_word);
        strncpy(word, argv[1] + i, len_word);
        word[len_word] = '\0';
    }
    else {
        len_word = 7;
        word = malloc(len_word * sizeof(char));
        if (!word || len_word <= 0) {
            fprintf(stderr, "Memory allocation failed for word\n");
            return -1;
        }
        strcpy(word, "random");
    }
    printf("word: %s\n", word);
    
    char lambda_1[10];
    sprintf(lambda_1, "%.1f", (float)LAMBDA_1); 
    char lambda_2[10];
    sprintf(lambda_2, "%.1f", (float)LAMBDA_2);

    char word_lambda_dot_csv[256];
    sprintf(word_lambda_dot_csv, "%s_%s_%s.csv", word, lambda_1, lambda_2);

    char *ista_file_name = malloc((len_word + 100) * sizeof(char));
    strcpy(ista_file_name, "results/ista_");
    strcat(ista_file_name, word_lambda_dot_csv);
    FILE *ista_file = fopen(ista_file_name, "w");

    char *fista_file_name = malloc((len_word + 100) * sizeof(char));
    strcpy(fista_file_name, "results/fista_");
    strcat(fista_file_name, word_lambda_dot_csv);
    FILE *fista_file = fopen(fista_file_name, "w");

    char *lbfgs_file_name = malloc((len_word + 100) * sizeof(char));
    strcpy(lbfgs_file_name, "results/lbfgs_");
    strcat(lbfgs_file_name, word_lambda_dot_csv);
    FILE *lbfgs_file;
    if (LAMBDA_1 == 0) lbfgs_file = fopen(lbfgs_file_name, "w");

    char *lbfgs_fista_file_name = malloc((len_word + 100) * sizeof(char));
    strcpy(lbfgs_fista_file_name, "results/lbfgs_fista_");
    strcat(lbfgs_fista_file_name, word_lambda_dot_csv);
    FILE *lbfgs_fista_file = fopen(lbfgs_fista_file_name, "w");

    free(word);
    free(ista_file_name);
    free(fista_file_name);
    free(lbfgs_file_name);
    free(lbfgs_fista_file_name);

    x_ista       = ista(x_ista, problem, ista_file);
    x_fista      = fista(x_fista, problem, fista_file);
    if (LAMBDA_1 == 0) x_lbfgs      = L_BFGS(x_lbfgs, 5, problem, lbfgs_file);
    x_lbfg_fista = LBFGS_fista(x_lbfg_fista, 5, problem, lbfgs_fista_file);

    fclose(ista_file);
    fclose(fista_file);
    if (LAMBDA_1 == 0) fclose(lbfgs_file);
    fclose(lbfgs_fista_file);

    /* --------------------------- clean‑up --------------------------- */
    free(x_ista);       free(x_fista);
    free(x_lbfgs);      free(x_lbfg_fista);

    if (A) {
        free(A[0]); /* contiguous block if loaded from CSV */
        free(A);
    }
    free(b);

    return 0;
}
