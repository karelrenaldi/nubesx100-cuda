#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATAMAX 1000
#define DATAMIN -1000

int *input_matrix(int *matrix, int n_elements)
{
    for (int i = 0; i < n_elements; i++)
    {
        scanf("%d", &matrix[i]);
    }
}

void print_matrix(int *matrix, int n_elements)
{
    for (int i = 0; i < n_elements; i++)
    {
        printf("%i: %d\n", i, matrix[i]);
    }
}

int main()
{
    int kernel_row, kernel_col, target_row, target_col, num_targets;

    // Receive kernel matrix inputs
    scanf("%d %d", &kernel_row, &kernel_col);
    int kernel_elements = kernel_row * kernel_col;

    int kernel_matrix[kernel_elements];
    input_matrix(kernel_matrix, kernel_elements);

    // Receive input matrices
    scanf("%d %d %d", &num_targets, &target_row, &target_col);
    int target_elements = num_targets * target_row * target_col;

    int target_matrices[target_elements];
    input_matrix(target_matrices, target_elements);

    print_matrix(kernel_matrix, kernel_elements);
    print_matrix(target_matrices, target_elements);
}