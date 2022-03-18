% % writefile test.cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATAMAX 1000
#define DATAMIN -1000
#define BLOCK_SIZE 1024

    void
    input_matrix(int *matrix, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
    {
        scanf("%d", &matrix[i]);
    }
}

void print_matrix(int *matrix, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
    {
        printf("%i: %d\n", i, matrix[i]);
    }
}

int ceil_division(int a, int b)
{
    return a / b + (a % b != 0);
}

__device__ void change_idx_1d_to_2d(int idx, int ncol, int *row, int *col)
{
    *row = idx / ncol;
    *col = idx % ncol;
}

__device__ int change_idx_2d_to_1d(int ncol, int row, int col)
{
    return ncol * row + col;
}

__global__ void convolution(
    int *d_output,
    int *d_matrix,
    int *d_kernel,
    int output_row,
    int output_col,
    int matrix_row,
    int matrix_col,
    int kernel_row,
    int kernel_col)
{
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= output_row * output_col)
    {
        return;
    }

    int offset = blockIdx.y * matrix_row * matrix_col;

    int output_i, output_j;
    change_idx_1d_to_2d(output_idx, output_col, &output_i, &output_j);

    int res = 0;
    int curr_kernel_idx, curr_matrix_idx;
    for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
    {
        for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
        {
            curr_kernel_idx = change_idx_2d_to_1d(kernel_col, kernel_i, kernel_j);
            curr_matrix_idx = change_idx_2d_to_1d(
                                  matrix_col,
                                  kernel_i + output_i,
                                  kernel_j + output_j) +
                              offset;

            res += d_kernel[curr_kernel_idx] * d_matrix[curr_matrix_idx];
        }
    }

    d_output[output_idx + (blockIdx.y * output_row * output_col)] = res;
}

__global__ void find_range(
    int *d_range_output,
    int *d_conv_input,
    int conv_row,
    int conv_col,
    int num_output,
    int datamax,
    int datamin)
{
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= num_output)
    {
        return;
    }

    int curr_max = datamin, curr_min = datamax;
    int num_conv_elements = conv_row * conv_col;
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * num_conv_elements;

    for (int i = offset; i < num_conv_elements + offset; i++)
    {
        int curr_el = d_conv_input[i];
        if (curr_el > curr_max)
        {
            curr_max = curr_el;
        }
        if (curr_el < curr_min)
        {
            curr_min = curr_el;
        }
    }

    d_range_output[output_idx] = curr_max - curr_min;
}

int main()
{
    int kernel_row, kernel_col;
    scanf("%d %d", &kernel_row, &kernel_col);

    int num_kernel_elements = kernel_row * kernel_col;
    int kernel_size = num_kernel_elements * sizeof(int);

    int *kernel, *d_kernel;
    kernel = (int *)malloc(kernel_size);
    cudaMalloc((void **)&d_kernel, kernel_size);

    input_matrix(kernel, num_kernel_elements);
    cudaError errKernel = cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
    if (errKernel != cudaSuccess)
    {
        printf("Failed to copy kernel to GPU\n");
    }

    int matrix_row, matrix_col, num_matrix;
    scanf("%d %d %d", &num_matrix, &matrix_row, &matrix_col);

    int num_matrix_elements = num_matrix * matrix_row * matrix_col;
    int matrix_size = num_matrix_elements * sizeof(int);

    int *matrix, *d_matrix;
    matrix = (int *)malloc(matrix_size);
    cudaMalloc((void **)&d_matrix, matrix_size);

    input_matrix(matrix, num_matrix_elements);
    cudaError errMatrix = cudaMemcpy(d_matrix, matrix, matrix_size, cudaMemcpyHostToDevice);
    if (errMatrix != cudaSuccess)
    {
        printf("Failed to copy matrix to GPU\n");
    }

    int conv_output_row = matrix_row - kernel_row + 1;
    int conv_output_col = matrix_col - kernel_col + 1;
    int num_output_elements = num_matrix * conv_output_row * conv_output_col;
    int conv_output_size = num_output_elements * sizeof(int);

    int *conv_output, *d_output;
    conv_output = (int *)malloc(conv_output_size);
    cudaMalloc((void **)&d_output, conv_output_size);

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(ceil_division(num_output_elements, BLOCK_SIZE), num_matrix);

    convolution<<<grid_dim, block_dim>>>(
        d_output,
        d_matrix,
        d_kernel,
        conv_output_row,
        conv_output_col,
        matrix_row,
        matrix_col,
        kernel_row,
        kernel_col);

    // Get convolution output from GPU.
    cudaMemcpy(conv_output, d_output, conv_output_size, cudaMemcpyDeviceToHost);

    int *d_conv_input;
    cudaMalloc((void **)&d_conv_input, conv_output_size);
    cudaError errConv = cudaMemcpy(d_conv_input, conv_output, conv_output_size, cudaMemcpyHostToDevice);
    if (errConv != cudaSuccess)
    {
        printf("Failed to copy convolution result to GPU\n");
    }

    int *range_output, *d_range_output;
    int range_output_size = num_matrix * sizeof(int);
    range_output = (int *)malloc(range_output_size);
    cudaMalloc((void **)&d_range_output, range_output_size);

    find_range<<<ceil_division(num_matrix, BLOCK_SIZE), block_dim>>>(
        d_range_output,
        d_conv_input,
        conv_output_row,
        conv_output_col,
        num_matrix,
        DATAMAX,
        DATAMIN);

    // Get range output from GPU.
    cudaMemcpy(range_output, d_range_output, range_output_size, cudaMemcpyDeviceToHost);
    print_matrix(range_output, num_matrix);

    // Cleanup.
    cudaFree(d_kernel);
    cudaFree(d_matrix);
    cudaFree(d_output);
    cudaFree(d_conv_input);
    cudaFree(d_range_output);
    free(kernel);
    free(matrix);
    free(conv_output);
    free(range_output);
}