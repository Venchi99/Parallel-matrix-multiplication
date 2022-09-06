# Parallel matrix multiplication
Task 1: Implement a parallel version of blocked matrix multiplication by OpenMP.<br>
Task 2: Implement SUMMA algorithm by MPI.<br>
Task 3: Implement Cannon’s algorithm by MPI.

Each task have two variants of implementation.
The code for Tasks 1-3 can accept external testcase input files. For example, Task1 accept following command line:
```
Task1 <num_threads> <input_testcase_filename> <output_filename>
```

In addition, dot_product_CUDA folder contains CUDA implementation of dot product. 
It contains four variants of implementation, and you can find performance report 
that compared block scheduling and cyclic scheduling with various vector/grid/block size.


