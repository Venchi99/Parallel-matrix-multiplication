#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// make task_11
// ./task_11 16 testcases/task_11_in test_case/task_11_out
// ./task_11 2 testcases/input_testcase64 testcases/task_11_out



int main(int argc, char *argv[]) {


    if (argc != 4) {
        printf("Incorrect number of arguments, please follow: ./task_11 <num_threads> <input_testcase_filename> <output_filename>\n");
        return -1;
    }

    int np = atoi(argv[1]);
    if (np < 1) {
        printf("Error: Number_of_threads  (%i) < 1 \n", np);
        return -1;
    }

    //read from input file, assume it will work properly
    FILE *fp;
    int r_a, c_a, r_b, c_b;
    int **a = NULL;
    int **b = NULL;
    int **c = NULL;

    fp = fopen(argv[2], "r");
    if (fp == NULL) {
        printf("Cannot open input file!\n");
        return -1;
    }
    fscanf(fp, "%d", &c_a);
    fscanf(fp, "%d", &r_a);
    a = (int **) calloc(c_a, sizeof(int *));
    for (int i = 0; i < c_a; i++) {
        a[i] = (int *) calloc(r_a, sizeof(int));
    }


    for (int i = 0; i < c_a; i++) {
        for (int j = 0; j < r_a; j++) {

            fscanf(fp, "%d", &a[i][j]);
        }
    }

    fscanf(fp, "%d", &c_b);
    fscanf(fp, "%d", &r_b);
    b = (int **) calloc(c_b, sizeof(int *));
    for (int i = 0; i < c_b; i++) {
        b[i] = (int *) calloc(r_b, sizeof(int));
    }

    for (int i = 0; i < c_b; i++) {
        for (int j = 0; j < r_b; j++) {
            fscanf(fp, "%d", &b[i][j]);

        }
    }
    fclose(fp);


    //output matrix
    c = (int **) calloc(c_a, sizeof(int *));
    for (int i = 0; i < c_a; i++) {
        c[i] = (int *) calloc(r_b, sizeof(int));
    }


    // If np > any dimension of two matrics, np will automatically decrease to this number.
    if (np > r_a) {
        np = r_a;
    }

    if (np > c_a) {
        np = c_a;
    }

    if (np > r_b) {
        np = r_b;
    }

    omp_set_num_threads(np);
    int block_n = omp_get_num_threads();

    double start = omp_get_wtime();


#pragma omp parallel shared(a, b, c, c_a, r_b, r_a, block_n)
    {
        int end_i;
        int end_j;
        int tmp;
        // calcluate end index for each thread
        if (omp_get_thread_num() == (omp_get_num_threads() - 1)) {
            end_i = c_a;
            end_j = r_b;

        } else {
            end_i = (1 + block_n) * omp_get_thread_num();
            end_j = (1 + block_n) * omp_get_thread_num();
        }
#pragma omp parallel for collapse(2)
        for (int i = omp_get_thread_num() * block_n; i < end_i; i++) {
            for (int j = omp_get_thread_num() * block_n; j < end_j; j++) {
                tmp = 0;
                for (int x = 0; x < r_a; x++) {
                    tmp += a[i][x] * b[x][j];
                }
                c[i][j] = tmp;
            }

        }
    }

    double end = omp_get_wtime();
    //write to output file
    FILE *f;
    f = fopen(argv[3], "w");
    if (f == NULL) {
        printf("Cannot open output file!\n");
        return -1;
    }

    for (int i = 0; i < c_a; i++) {
        for (int j = 0; j < r_b; j++) {
            fprintf(f, "%d ", c[i][j]);
        }
        fprintf(f, "\n");
    }

    fprintf(f, "Running time = %fms\n", (end - start) * 1000);
    fclose(f);

    for (int i = 0; i < c_a; i++) {
        free(c[i]);
        free(a[i]);
    }
    // free allocated memory
    for (int i = 0; i < c_b; i++) {
        free(b[i]);

    }
    free(a);
    free(b);
    free(c);
    return 0;
}
