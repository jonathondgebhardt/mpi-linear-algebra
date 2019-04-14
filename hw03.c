#include "nehrbass.h"

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/*
 * Leverage Slurm and MPI to solve the linear equation Ax = Y, where A and Y are
 * square matrices and x is a column matrix.
 *
 * Author: Jonathon Gebhardt
 * Class: CS4900-B90
 * Instructor: Dr. John Nehrbass
 * Assignment: Homework 3
 * GitHub: https://github.com/jonathondgebhardt/mpi-linear-algebra
 */

void showUsage(char*);
void print2dMatrix(double**, int);
void print1dMatrix(double*, int);
double** createDiagonalMatrix(int);
double** createRandomMatrix(int);
int getDimensionFromFile(char*);
double** getMatrixFromFile(char*);
double* createColumnMatrix(int);

int main(int argc, char* argv[])
{
    int rank, ncpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    int opt, rFlag = 0, dFlag = 0, fFlag = 0, dimension;
    char* fileName;

    while ((opt = getopt(argc, argv, "r:d:f:")) != -1)
    {
        switch (opt)
        {
            case 'r':
                rFlag = 1;
                dimension = atoi(optarg);
                break;
            case 'd':
                dFlag = 1;
                dimension = atoi(optarg);
                break;
            case 'f':
                fFlag = 1;
                fileName = optarg;
                break;
            case '?':
                if (rank == 0)
                {
                    showUsage(argv[0]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
        }
    }

    if (rank == 0)
    {
        // Validate user input.
        if (rFlag == 0 && dFlag == 0 && fFlag == 0)
        {
            fprintf(stderr,
                    "A value for -r, -d, or -f exclusively is required\n");
            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        else if (rFlag + dFlag + fFlag > 1)
        {
            fprintf(stderr, "Only one option may be used at a time\n");
            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Seeding rand() isn't necessary for every case, but seed it here once to
    // simplify logic.
    srand(time(NULL));

    // Populate matrix based on user request.
    double** matrix;
    if (rFlag == 1)
    {
        matrix = createRandomMatrix(dimension);
    }
    else if (dFlag == 1)
    {
        matrix = createDiagonalMatrix(dimension);
    }
    else
    {
        matrix = getMatrixFromFile(fileName);
        if (matrix == NULL)
        {
            fprintf(stderr, "Error reading from file\n");
            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        dimension = getDimensionFromFile(fileName);
    }

    printf("\nThe given matrix is:\n");
    print2dMatrix(matrix, dimension);

    // If the determinant is 0, we can't do any meaningful work.
    double det;
    if ((det = determinantNehrbass(matrix, 0, dimension, dimension)) == 0)
    {
        printf("Determinant is zero, infinitely many solutions exist\n");
    }
    else
    {
        printf("\nDeterminant of the given matrix is:\n %7.2f\n", det);

        double** cofactorMatrix = cofactor(matrix, dimension);
        double** inverse = transpose(matrix, cofactorMatrix, dimension);

        printf("\nInverse of the given matrix is:\n");
        print2dMatrix(inverse, dimension);

        // Generate a random 'x' to solve Ax = Y.
        double* sampleX = createColumnMatrix(dimension);
        printf("\nThe generated sampleX is:\n");
        print1dMatrix(sampleX, dimension);

        double* yMatrix = create1dDoubleMatrix(dimension);
        int i;
        for (i = 0; i < dimension; ++i)
        {
            yMatrix[i] = dot(matrix[i], sampleX, dimension);
        }

        printf("\nThe computed yMatrix is:\n");
        print1dMatrix(yMatrix, dimension);

        // Clean up.
        free(sampleX);
        free(yMatrix);

        for (i = 0; i < dimension; ++i)
        {
            free(cofactorMatrix[i]);
            free(inverse[i]);
        }
        free(cofactorMatrix);
        free(inverse);
    }

    // Clean up.
    if (matrix != NULL)
    {
        int i;
        for (i = 0; i < dimension; ++i)
        {
            free(matrix[i]);
        }

        free(matrix);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}

void showUsage(char* applicationName)
{
    printf("Usage: %s [-r m] [-d n] [-f fileName]\n", applicationName);
    printf("\t-r: m x m matrix filled with random numbers\n");
    printf(
        "\t-d: n x n diagonal matrix where the values are the row numbers\n");
    printf(
        "\t-f: read a matrix from a file where the first line is the square "
        "dimension and the second line contains space delimited elements\n");
}

void print2dMatrix(double** arr, int dimension)
{
    int i;
    for (i = 0; i < dimension; ++i)
    {
        int j;
        for (j = 0; j < dimension; ++j)
        {
            printf("%7.2f ", arr[i][j]);
        }

        printf("\n");
    }
}

void print1dMatrix(double* arr, int dimension)
{
    int i;
    for (i = 0; i < dimension; ++i)
    {
        printf("%7.2f ", arr[i]);
    }

    printf("\n");
}

double** createDiagonalMatrix(int dimension)
{
    double** arr = create2dDoubleMatrix(dimension);

    int i, j;
    for (i = 0; i < dimension; ++i)
    {
        for (j = 0; j < dimension; ++j)
        {
            if (i == j)
            {
                arr[i][j] = j + 1;
            }
            else
            {
                arr[i][j] = 0;
            }
        }
    }

    return arr;
}

double** createRandomMatrix(int dimension)
{
    double** arr = create2dDoubleMatrix(dimension);

    int i, j, ceiling = 100;
    for (i = 0; i < dimension; ++i)
    {
        for (j = 0; j < dimension; ++j)
        {
            arr[i][j] = (double)rand() / ((double)RAND_MAX + 1) * ceiling;
        }
    }

    return arr;
}

int getDimensionFromFile(char* fileName)
{
    int dimension = 0;

    FILE* fp = fopen(fileName, "r");
    if (fp != NULL)
    {
        fscanf(fp, "%d", &dimension);
        assert(dimension != 0);

        fclose(fp);
    }

    return dimension;
}

double** getMatrixFromFile(char* fileName)
{
    double** arr = NULL;

    FILE* fp = fopen(fileName, "r");
    if (fp != NULL)
    {
        // First line contains dimension (n x n).
        int dimension;
        fscanf(fp, "%d", &dimension);
        assert(dimension != 0);

        // Next line contains matrix contents (space delimited doubles).
        arr = create2dDoubleMatrix(dimension);

        int i, j;
        for (i = 0; i < dimension; ++i)
        {
            for (j = 0; j < dimension; ++j)
            {
                fscanf(fp, "%lf", &arr[i][j]);
            }
        }

        fclose(fp);
    }

    return arr;
}

double* createColumnMatrix(int dimension)
{
    double* arr = create1dDoubleMatrix(dimension);

    int i, ceiling = 100;
    for (i = 0; i < dimension; ++i)
    {
        arr[i] = (double)rand() / ((double)RAND_MAX + 1) * ceiling;
    }

    return arr;
}
