#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <string.h>

// cd code
// module load openmpi
// mpicc -g -Wall -lm -o task_22 PartA/task_22.c
// mpirun -n 4 ./task_22 testcases/input_testcase5.txt testcases/task_22_out
// mpirun -n 16 ./task_22 testcases/input_testcase_large2 testcases/task_22_out

int main(int argc, char *argv[]) {
    double start, end;
    int r_a, c_a, r_b, c_b;
    int *a;
    int *b;
    int *c;
    int *local_a, *local_b, *local_c, *rece_a, *rece_b;


    //start up MPI
    MPI_Init(&argc, &argv);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Status status;

    if (my_rank == 0) {


        //read from input file, assume it will work properly

        FILE *fp;


        fp = fopen(argv[1], "r");
        if (fp == NULL) {
            printf("Cannot open input file!\n");
            return -1;
        }
        fscanf(fp, "%d", &c_a);
        fscanf(fp, "%d", &r_a);
        a = (int *) malloc(c_a * r_a * sizeof(int));


        for (int i = 0; i < c_a * r_a; i++) {
            fscanf(fp, "%d", &a[i]);
        }

        fscanf(fp, "%d", &c_b);
        fscanf(fp, "%d", &r_b);

        b = (int *) malloc(c_b * r_b * sizeof(int));

        for (int i = 0; i < c_b * r_b; i++) {
            fscanf(fp, "%d", &b[i]);
        }
        fclose(fp);
        start = MPI_Wtime();
        //printf("from 66: %d\n", my_rank);
    }

    MPI_Bcast(&c_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int p = (int) sqrt(comm_size);

    int start_a[comm_size];
    int start_b[comm_size];


    int c_a_local = (int) c_a / p;
    int c_b_local = (int) c_b / p;
    int r_a_local = (int) r_a / p;
    int r_b_local = (int) r_b / p;

    MPI_Datatype my_type;
    MPI_Type_vector(c_a_local, r_b_local, c_a, MPI_INT, &my_type);   ////
    MPI_Type_commit(&my_type);


    local_a = (int *) malloc(c_a_local * r_a_local * sizeof(int));
    local_b = (int *) malloc(c_b_local * r_b_local * sizeof(int));
    local_c = (int *) malloc(c_a_local * r_b_local * sizeof(int));

    rece_a = (int *) malloc(c_a_local * r_a_local * sizeof(int));
    rece_b = (int *) malloc(c_b_local * r_b_local * sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);
    //printf("from 92: %d\n", c_a_local*r_b_local);
    memset(local_c, 0, c_a_local * r_b_local * sizeof(int));

    int index = 0;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            start_a[index] = i * r_a * c_a_local + j * r_a_local;
            start_b[index] = i * r_b * c_b_local + j * r_b_local;
            index++;
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0) {
        //process 0 send all submatrixs to corresponding process.
        int counter = 0;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                int cursor = 0;
                for (int x = 0; x < c_a_local; x++) {
                    for (int y = 0; y < r_a_local; y++) {
                        rece_a[cursor] = a[start_a[counter] + x * r_a + y];
                        cursor++;
                    }
                }
                //blocking
                MPI_Send(&rece_a[0], r_a_local * c_a_local, MPI_INT, counter, counter + 860, MPI_COMM_WORLD);
                cursor = 0;
                for (int x = 0; x < c_b_local; x++) {
                    for (int y = 0; y < r_b_local; y++) {
                        rece_b[cursor] = b[start_b[counter] + x * r_b + y];
                        cursor++;
                    }
                }
                //blocking
                MPI_Send(&rece_b[0], r_b_local * c_b_local, MPI_INT, counter, counter + 500, MPI_COMM_WORLD);
                counter++;
            }

        }

    }


    MPI_Barrier(MPI_COMM_WORLD);
    // blocking
    MPI_Recv(&local_a[0], r_a_local * c_a_local, MPI_INT, 0, my_rank + 860, MPI_COMM_WORLD, &status);
    MPI_Recv(&local_b[0], r_b_local * c_b_local, MPI_INT, 0, my_rank + 500, MPI_COMM_WORLD, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    memset(rece_a, 0, c_a_local * r_a_local * sizeof(int));
    memset(rece_b, 0, c_b_local * r_b_local * sizeof(int));

    // Split row/column communicators for current process.
    int row_c = my_rank / p;
    int col_c = my_rank % p;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_c, my_rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_c, my_rank, &col_comm);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < p; i++) {
        if (col_c == i) {
            memcpy(rece_a, local_a, sizeof(int) * c_a_local * r_a_local);
        }

        if (row_c == i) {
            memcpy(rece_b, local_b, sizeof(int) * c_b_local * r_b_local);


        }
        //owner of A[i, k] broadcasts it to all processors in the same row.
        MPI_Bcast(&rece_a[0], c_a_local * r_a_local, MPI_INT, i, row_comm);
        //owner of b[k, j] broadcasts it to all processors in the same column.
        MPI_Bcast(&rece_b[0], c_b_local * r_b_local, MPI_INT, i, col_comm);

        // matrix mutiplication
        for (int x = 0; x < c_a_local; x++) {
            for (int y = 0; y < r_b_local; y++) {
                int tmp = 0;
                for (int z = 0; z < r_a_local; z++) {
                    tmp += rece_a[x * r_a_local + z] * rece_b[z * r_b_local + y];

                }

                local_c[x * r_b_local + y] += tmp;

            }
        }

    }


    MPI_Barrier(MPI_COMM_WORLD);
    // send local c to process 0, blocking
    MPI_Send(&local_c[0], c_a_local * r_b_local, MPI_INT, 0, my_rank + 356, MPI_COMM_WORLD);






    if (my_rank == 0) {
        // receive all local_c from different process and collected at c, blocking
        c = (int *) malloc(c_a * r_b * sizeof(int));
        int counter = 0;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                MPI_Recv(&c[r_b * i * r_b_local + j * c_a_local], 1, my_type, counter, counter + 356, MPI_COMM_WORLD,
                         &status);
                counter++;
            }

        }

        end = MPI_Wtime();
        //write to output file
        FILE *f;
        f = fopen(argv[2], "w");
        if (f == NULL) {
            printf("Cannot open output file!\n");
            return -1;
        }

        for (int i = 0; i < c_a; i++) {
            for (int j = 0; j < r_b; j++) {
                fprintf(f, "%d ", c[i * r_b + j]);
            }
            fprintf(f, "\n");
        }

        fprintf(f, "Running time = %fms\n", (end - start) * 1000);
        fclose(f);
        //printf("from 205: %d\n", my_rank);
        free(a);
        free(b);
        free(c);

    }
    free(local_a);
    free(local_b);
    free(rece_a);
    free(rece_b);
    free(local_c);
    //printf("from 216: %d\n", my_rank);

    MPI_Finalize();
    return 0;

}