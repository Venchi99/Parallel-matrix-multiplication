
## Compare the performance of Kernel with block distribution of array indices and Kernel with cyclic distribution of array indices
### Experimental Procedure

Firstly, consider various vector/grid/block size as three different scenarios:
1. total number of thread is smaller than array size
2. total number of thread is larger than array size
3. total number of thread is equal to array size

With these three different scenarios, we also consider different scales and whether vector size is divisible for gird/block size.
Hence, I designed following combination of vector/grid/block size:
|vector size|grid size|block size|
|---|---|---|
| 4  | 2  | 2 |
| 500  | 100 | 5  |
| 50 | 3 | 3  |
| 1000  |  6 |  9 |
| 31  |  28 |  6 |
|  73 |  3 |  111 |
| 120000 |50|721|





### Results
| vector size | grid size | block size | block distribution(seconds) | cyclic distribution(seconds) |
|-------------|---------|------------|-----------|:----------------------------:|
| 4           | 2       | 2          | 1.907349e-05 |         1.406670e-05         |
| 500         | 100     | 5          |1.692772e-05 |         1.597404e-05         |
| 50          | 3       | 3          |  1.502037e-05        |         1.692772e-05         |
| 1000        | 6       | 9          |   2.002716e-05       |         1.907349e-05         |
| 31          | 28      | 6          |   2.288818e-05         |         2.002716e-05         |
| 73          | 3       | 111        |   1.883507e-05        |         1.597404e-05         |
| 120000      | 50      | 721        |   4.639626e-04      |         1.010895e-04         |



### Performance explanation
According to the results, block distribution only performed slightly better on the third testcase. 
While, cyclic distribution performs better than block distribution in most cases, especially the last one, more than 4 times faster.
Hence, it seems that cyclic distribution overall have better performance compare to block distribution. 