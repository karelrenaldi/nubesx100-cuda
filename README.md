# IF3230-K04-nubesx100-CUDA

Matrix Convolution with CUDA

## CUDA Parallelization Scheme

### Convolution Calculation

For the first stage of convolution calculation, a 2D grid and a 1D block will be created first, where the x-axis grid value is the total number of convolutions divided by the static value BLOCK_SIZE of 1024, and the y-axis grid value is the total number of input matrices. Since the block itself is 1-dimensional, the x-axis block value is statically set to BLOCK_SIZE, which is 1024. Broadly speaking, the scheme created ensures that one thread calculates the result of one convolution. The main function will receive all input matrices, the kernel, and the output matrix along with all their sizes as parameters. Inside the function, the output index will be determined, where the convolution result will be stored. Two nested loops will iterate over the kernel and compute the convolution results. As additional information, all input, kernel, and output matrices will be represented in a 1D format.

### Data Range Search

The data range search is performed using a 1D array generated from the convolution calculation. On both the GPU and CPU, memory will be allocated to store the results of the data range search for the number of input matrices provided by the user. A grid will then be used, sized as the ceiling of the matrix size divided by BLOCK_SIZE, with the same block dimension. Each thread will sequentially fill an entry in the output array `(0, 1, ..., n_matrix - 1)`. Here, the initial offset for each thread will be determined, and using this offset, the thread will perform a serial search for the maximum and minimum values from that offset up to the number of convolution result elements for a matrix.

### Sorting

Sorting is performed in a maximum of two stages. In the first stage, merge sort is executed using the number of threads equal to BLOCK_SIZE and the number of blocks equal to the ceiling of num_matrix divided by BLOCK_SIZE. Each block handles BLOCK_SIZE elements, and one thread performs a merge sort between two arrays. Each block undergoes a maximum of log2(BLOCK_SIZE) passes, where the first pass uses a maximum of BLOCK_SIZE threads to sort a total of BLOCK_SIZE arrays, each of size 1 element. The second pass uses a maximum of BLOCK_SIZE/2 threads to sort a total of BLOCK_SIZE/2 arrays of size 2 elements, and so on. In the second stage, merge sort is performed to combine the sorted results of each block. This second stage is only executed if the number of blocks in the first stage is more than one. In the second stage, only one block exists, with a maximum number of threads equal to the number of blocks in the first stage. In the first pass, a maximum of N_BLOCK threads merges two arrays of size BLOCK_SIZE. In the second pass, a maximum of the number of blocks divided by 2 threads merges two arrays of size BLOCK_SIZE * 2, and so on.

## Best Execution Analysis

For each test case, the parallel program executes faster than the serial program. The speedup for TC1 is 7.514x the serial execution, for TC2 is 54.698x, for TC3 is 22.376x, and for TC4 is 22.250x. The best speedup is achieved in test case 2, with test parameters of an 18x12 kernel size and 101 matrices of size 100x100. This is due to a good level of parallelization, reducing the 1.64*10^8 serial convolution operations to 216 operations per thread. In TC2, the overhead of block creation is also not as high as in TC3 or TC4. In TC2, 101 blocks need to be created on the y-axis grid and 8 blocks on the x-axis grid. Compared to 666 blocks on the y-axis of TC3 and 5000 blocks on the y-axis of TC4, the overhead for TC2 is significantly lower. Therefore, TC2 has the best speedup among all test cases.

## Comparison of Serial and Parallel Execution Results

By comparing the results of the serially executed program with those of the parallel execution, it is proven that both programs produce the same results. This confirms that the parallel program successfully addresses potential issues in parallel applications and has been correctly implemented. Additionally, as shown in the table below, the parallel program significantly improves performance, with a rate that increases as the problem size grows. Thus, it can be concluded that the parallel program has been well-designed and correctly implemented.

## Execution Variation Experiments

```shell
| TC  | CUDA (seconds) | Serial (seconds) |
| --- | ------------- | --------------- |
| 1   | 0.001277      | 0.009592        |
| 2   | 0.013848      | 0.757452        |
| 3   | 0.032927      | 0.736760        |
| 4   | 0.438060      | 9.746812        |
| a   | 0.002916      | -               |
| b   | 0.000182      | -               |
| c   | 0.001897      | -               |
```

a = 10,000 matrices, with kernel and matrix size each being 1x1 and 1x1.
b = 1 matrix, with kernel and matrix size each being 1x1 and 100x100.
c = 1 matrix, with kernel and matrix size each being 100x100 and 100x100.

For the three additional cases, it is observed that the second case has the fastest execution time, followed closely by the third case, and finally the first case, which takes the longest. This is because the first case creates 10,000 blocks on the y-axis grid, resulting in high overhead. The third case takes longer than the second because, in each thread, the third case handles 100*100 operations, while the second case only handles 1.

## Author

1. 13519180 Karel Renaldi
2. 13519185 Richard Rivaldo
3. 13519205 Muhammad Rifat Abiwardani
