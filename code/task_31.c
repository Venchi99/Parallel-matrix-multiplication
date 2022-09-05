#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <string.h>


// mpicc -g -Wall -lm -o task_31 PartA/task_31.c
// mpirun -n 16 ./task_31 testcases/input_testcase1000 testcases/task_31_out
int main(int argc, char *argv[]) {

    FILE *fp;
    int *a, *b, *c, *local_a, *local_b, *local_c;
    int r_a, c_a, r_b, c_b, N, local_n;
    double start, end;
    start = MPI_Wtime();
    MPI_Init(&argc, &argv);
    int comm_size;
    int periods[] = {1, 1};
    int dim[2];
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int my_rank, my_coords[2];
    int up, down, left, right, source, dest;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm comm;
    MPI_Status status;


    if (my_rank == 0) {
        //read from input file, assume it will work properly
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

        N = c_a;
        c = (int *) malloc(c_a * r_b * sizeof(int));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    dim[0] = dim[1] = (int) sqrt(comm_size);
    local_n = (int) N / dim[0];

    // create new MPI data type for easier send/receive data
    MPI_Datatype my_type;
    MPI_Type_vector(local_n, local_n, N, MPI_INT, &my_type);
    MPI_Type_commit(&my_type);


    local_a = (int *) malloc(local_n * local_n * sizeof(int));
    local_b = (int *) malloc(local_n * local_n * sizeof(int));
    local_c = (int *) malloc(local_n * local_n * sizeof(int));
    memset(local_c, 0, sizeof(int) * local_n * local_n);


    if (my_rank == 0) {
        // split and send a, b to each process from process 0
        int counter = 0;
        for (int i = 0; i < dim[0]; i++) {
            for (int j = 0; j < dim[0]; j++) {
                MPI_Send(&a[(i * local_n * N) + (j * local_n)], 1, my_type, counter, 600 + counter, MPI_COMM_WORLD);
                MPI_Send(&b[(i * local_n * N) + (j * local_n)], 1, my_type, counter, 1200 + counter, MPI_COMM_WORLD);
                counter++;
            }
        }

    }
    // receive
    MPI_Recv(&local_a[0], local_n * local_n, MPI_INT, 0, 600 + my_rank, MPI_COMM_WORLD, &status);
    MPI_Recv(&local_b[0], local_n * local_n, MPI_INT, 0, 1200 + my_rank, MPI_COMM_WORLD, &status);
    MPI_Barrier(MPI_COMM_WORLD);


    //cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 1, &comm);
    MPI_Cart_coords(comm, my_rank, 2, my_coords);

    MPI_Cart_shift(comm, 1, 1, &left, &right);
    MPI_Cart_shift(comm, 0, 1, &up, &down);

    // "skew" A
    MPI_Cart_shift(comm, 1, my_coords[0], &source, &dest);
    MPI_Sendrecv_replace(&local_a[0], local_n * local_n, MPI_INT, source, 1, dest, 1, comm, &status);

    // "skew" B
    MPI_Cart_shift(comm, 0, my_coords[1], &source, &dest);
    MPI_Sendrecv_replace(&local_b[0], local_n * local_n, MPI_INT, source, 1, dest, 1, comm, &status);


    for (int i = 0; i < dim[0]; i++) {
        // matrix mutiplication
        for (int x = 0; x < local_n; x++) {
            for (int y = 0; y < local_n; y++) {
                for (int z = 0; z < local_n; z++) {
                    local_c[x * local_n + y] += local_a[x * local_n + z] * local_b[z * local_n + y];
                }
            }
        }

        //blocking send & receive
        // left-circular-shift each row of A by 1
        MPI_Sendrecv_replace(&local_a[0], local_n * local_n, MPI_INT, left, 1, right, 1, comm, &status);
        // up-circular-shift each column of B by 1
        MPI_Sendrecv_replace(&local_b[0], local_n * local_n, MPI_INT, up, 1, down, 1, comm, &status);
        //printf("from %d hello 178\n", my_rank);   


    }

    //send to process 0 
    MPI_Send(&local_c[0], local_n * local_n, MPI_INT, 0, my_rank + 356, MPI_COMM_WORLD);


    MPI_Comm_free(&comm);
    free(local_a);
    free(local_b);

    //write to output file
    if (my_rank == 0) {
        // collect all local_c
        int counter = 0;
        for (int x = 0; x < dim[0]; x++) {
            for (int y = 0; y < dim[0]; y++) {
                MPI_Recv(&c[x * local_n * N + y * local_n], 1, my_type, counter, counter + 356, MPI_COMM_WORLD,
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
                fprintf(f, "%d ", c[i * c_a + j]);
            }
            fprintf(f, "\n");
        }

        fprintf(f, "Running time = %fms\n", (end - start) * 1000);
        fclose(f);
        free(a);
        free(b);
        free(c);

    }
    free(local_c);

    MPI_Finalize();

    return 0;

}