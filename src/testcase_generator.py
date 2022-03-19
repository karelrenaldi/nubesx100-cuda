import numpy as np

if __name__ == "__main__":
    file_name = input("Enter File Name: ")
    file = open(f"{file_name}.txt", "w")

    kernel_row = int(input("Enter Kernel Row Dimension: "))
    kernel_col = int(input("Enter Kernel Col Dimension: "))
    file.write(f"{kernel_row} {kernel_col}\n")

    random_kernel = np.random.randint(
        low=-1000, high=1001, size=(kernel_row, kernel_col)
    )
    for row in random_kernel:
        str_row = " ".join(row.astype(str)) + "\n"
        file.write(str_row)

    num_matrix = int(input("Enter Number of Matrices: "))
    matrix_row = int(input("Enter Matrix Row Dimension: "))
    matrix_col = int(input("Enter Matrix Col Dimension: "))

    file.write(f"{num_matrix} {matrix_row} {matrix_col}\n")

    for i in range(num_matrix):
        random_matrix = np.random.randint(
            low=-1000, high=1001, size=(matrix_row, matrix_col)
        )

        for row in random_matrix:
            str_row = " ".join(row.astype(str)) + "\n"
            file.write(str_row)

    file.close()
