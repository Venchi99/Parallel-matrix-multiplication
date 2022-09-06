#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define DEBUG

// nvcc -o dot_prod dot_prod.cu
// ./dot_prod 31 28 6 2b


static int n;
static float dot_parallel;
static float dot_serial;
#define THREADS_PER_BLOCK = 32 //only use at task 3


float Serial_dot_prod(float x[], float y[], int n) {//serial
    float cdot = 0.0;

    for (int i = 0; i < n; i++) {
        cdot += x[i] * y[i];
    }
    return cdot;
}

//GPU Kernel
__global__ void dot_1(float *a, float *b, float *c, int n)//basic parallel, assume thred num = vector size
{
    //Assuming n is the number of threads, using atomicAdd


    //compute each threads' corresponding array index.
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    //then, use atomicAdd to add the result to c; you can assume each thread will be responsible for only one element
    if (index < n) {
        atomicAdd(c, a[index] * b[index]);

    }

}

//GPU Kernel
__global__ void dot_2a(float *a, float *b, float *c, int n)//block distribution
{
    // 2a: block distribution

    // compute array elems per block
    int num_per_block, num_per_thread, start_index, end_index;
    if (n < gridDim.x) {
        num_per_block = 1;
    } else {
        num_per_block = n / gridDim.x;
    }

    // compute array elems per thread
    if (num_per_block < blockDim.x) {
        num_per_thread = 1;
    } else {
        num_per_thread = num_per_block / blockDim.x;
    }

    if (num_per_block == 1) {

    }
    start_index = num_per_thread * blockDim.x * blockIdx.x + num_per_thread * threadIdx.x;

    if (start_index >= n) {
        end_index = start_index;
    } else if ((blockIdx.x == (gridDim.x - 1)) && (threadIdx.x == (blockDim.x - 1))) {
        end_index = n;
    } else {
        end_index = num_per_thread * blockDim.x * blockIdx.x + num_per_thread * (threadIdx.x + 1);
    }

    float current = 0.0;
    for (int i = start_index; i < end_index; i++) {
        if (i >= n) {
            break;
        }
        current += a[i] * b[i];
    }
    atomicAdd(c, current);

    // loop through each thread's responsible task; achieve block distribution WITHIN each block.
    //Work for multiple blocks and when thread number != array length.


}

__global__ void dot_2b(float *a, float *b, float *c, int n)//cyclic distribution
{
    // 2b: cyclic distribution

    // compute array elems per block
    int total_num_thread = gridDim.x * blockDim.x;

    int num_per_block, num_per_thread;
    if (n < gridDim.x) {
        num_per_block = 1;
    } else {
        num_per_block = n / gridDim.x; //round down
    }

    // compute array elems per thread
    if (num_per_block < blockDim.x) {
        num_per_thread = 1;
    } else {
        num_per_thread = num_per_block / blockDim.x; //round down
    }

    int index = blockDim.x * blockIdx.x + threadIdx.x;


    float current = 0.0;
    //Use <= to have one more iteration, since previous steps always round down
    for (int i = 0; i <= num_per_thread; i++) {
        if (index >= n) {
            break;
        }
        current += a[index] * b[index];
        index += total_num_thread;

    }
    atomicAdd(c, current);

    // loop through each thread's responsible task; achieve cyclic distribution WITHIN each block.
    //Work for multiple blocks and when thread number != array length.

}


//GPU Kernel
__global__ void dot_3(float *a, float *b, float *c, int n)//shared memory
{
    //3: optimize with shared memory

    // allocate a shared array; they will be shared within block; assume threads per block is 32;
    __shared__ float my_shared_array[32];
    my_shared_array[threadIdx.x] = 0.0;

    __syncthreads();

    int total_num_thread = gridDim.x * blockDim.x;
    // compute array elems per block
    int num_per_block, num_per_thread;
    if (n < gridDim.x) {
        num_per_block = 1;
    } else {
        num_per_block = n / gridDim.x; //round down
    }

    // compute array elems per thread
    if (num_per_block < blockDim.x) {
        num_per_thread = 1;
    } else {
        num_per_thread = num_per_block / blockDim.x; //round down
    }

    int index = blockDim.x * blockIdx.x + threadIdx.x;


    //Use <= to have one more iteration, since previous steps always round down
    for (int i = 0; i <= num_per_thread; i++) {
        if (index >= n) {
            break;
        }
        my_shared_array[threadIdx.x] += a[index] * b[index];
        index += total_num_thread;

    }//loop through each thread's responsible task, utilize shared memory.
    __syncthreads();
    float current = 0.0;
    for (int i = 0; i < 32; i++) {
        current += my_shared_array[i];

        if ((threadIdx.x == 31) && (i == 31)) {
            atomicAdd(c, current);
        }
    }
   //one thread per block add the partial sum saved in shared memory to the result c.
    //Find a way to ensure every thread finishes before adding.

}


void Init_vectors(float x[], float y[], int n) {
    for (int i = 0; i < n; i++) {
        // Generate a random number
        float x_val = (float) rand() / RAND_MAX;
        float y_val = (float) rand() / RAND_MAX;
        // Put the random number into the correct array cell
        x[i] = x_val;
        y[i] = y_val;
    }
#ifdef DEBUG
    printf("X values:\n");
    for (int i = 0; i < n; i++) {
        printf("%f\n", x[i]);
    }
    printf("Y values:\n");
    for (int i = 0; i < n; i++) {
        printf("%f\n", y[i]);
    }
#endif
}

void Allocate_vectors(float **x_p, float **y_p, float **dot_p, int n) {
    cudaMallocManaged(x_p, n * sizeof(float));
    cudaMallocManaged(y_p, n * sizeof(float));
    cudaMallocManaged(dot_p, sizeof(float));

}

void Free_vectors(float *x, float *y, float *dot) {
    cudaFree(x);
    cudaFree(y);
    cudaFree(dot);
}


int main(int argc, char *argv[]) {
    int th_per_blk, blk_ct;
    if (argc != 5) {
        printf(" Number of params wrong\n");
        return -1;
    }
    n = atoi(argv[1]);
    blk_ct = atoi(argv[2]);
    th_per_blk = atoi(argv[3]);
    char *kernel = argv[4]; //kernel name: 1,2a,2b,3
    printf("%s\n", kernel);

    //declare variables cpu/device
    float x_cpu[n], y_cpu[n]; //n is static by cuda's need
    float *x_device, *y_device, *dot_device;
    Init_vectors(x_cpu, y_cpu, n);//random fill vectors (on cpu)

    //time serial computation
    double start_serial, finish_serial, elapsed_serial;
    double start, finish, elapsed;
    GET_TIME(start_serial);
    dot_serial = Serial_dot_prod(x_cpu, y_cpu, n);//serial dot product
    GET_TIME(finish_serial);
    elapsed_serial = finish_serial - start_serial;
    printf("Serial computation took %e seconds\n", elapsed_serial);

    //alloc cuda memory
    Allocate_vectors(&x_device, &y_device, &dot_device, n);

    //copy array to and from GPU
    cudaMemcpy(x_device, x_cpu, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_cpu, n * sizeof(float), cudaMemcpyHostToDevice);

    GET_TIME(start);
    //call kernel
    if (!strcmp(kernel, "1"))
        dot_1<<< blk_ct, th_per_blk>>>(x_device, y_device, dot_device, n);
    else if (!strcmp(kernel, "2a"))
        dot_2a<<< blk_ct, th_per_blk>>>(x_device, y_device, dot_device, n);
    else if (!strcmp(kernel, "2b"))
        dot_2b<<< blk_ct, th_per_blk>>>(x_device, y_device, dot_device, n);
    else if (!strcmp(kernel, "3"))
        dot_3<<< blk_ct, th_per_blk>>>(x_device, y_device, dot_device, n);
    else {
        printf("Kernel name not right \n");
        return -1;
    }
    cudaDeviceSynchronize();
    GET_TIME(finish);
    cudaMemcpy(&dot_parallel, dot_device, sizeof(float), cudaMemcpyDeviceToHost);
    elapsed = finish - start;
    printf("Parallel computation took %e seconds\n", elapsed);

    //save parallel result to `dot_parallel`

    if (fabs(dot_parallel - dot_serial) < 1e-3 * th_per_blk * blk_ct)
        printf("Result is CORRECT; parallel result: %f, serial result: %f \n", dot_parallel, dot_serial);
    else
        printf("Result is FALSE; parallel result: %f, serial result: %f \n", dot_parallel, dot_serial);

    //free cuda vectors
    Free_vectors(x_device, y_device, dot_device);

}
