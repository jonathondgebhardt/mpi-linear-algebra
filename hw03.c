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
 * Instructor: Dr. John Nerhbass
 * Assignment: Homework 3
 * GitHub: https://github.com/jonathondgebhardt/mpi-linear-algebra
 */

void showUsage(char*);
float** createDiagonalMatrix(int);
float** createRandomMatrix(int);
int getDimensionFromFile(char*);
float** readMatrixFromFile(char*);

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

    // Populate matrix based on user request.
    float** matrix;
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
        matrix = readMatrixFromFile(fileName);
        if (matrix != NULL)
        {
            dimension = getDimensionFromFile(fileName);
        }
        else
        {
            fprintf(stderr, "Error reading from file\n");
            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    if (matrix != NULL)
    {
        int i;
        for (i = 0; i < dimension; ++i)
        {
            free(matrix[i]);
        }

        free(matrix);
    }

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

float** createDiagonalMatrix(int dimension)
{
    float** arr = (float**)malloc(dimension * sizeof(float*));
    assert(arr != NULL);

    int i, j;
    for (i = 0; i < dimension; ++i)
    {
        arr[i] = (float*)malloc(dimension * sizeof(float));
        assert(arr[i] != NULL);

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

float** createRandomMatrix(int dimension)
{
    float** arr = (float**)malloc(dimension * sizeof(float*));
    assert(arr != NULL);

    srand(time(0));

    int i, j, ceiling = 100;
    for (i = 0; i < dimension; ++i)
    {
        arr[i] = (float*)malloc(dimension * sizeof(float));
        assert(arr[i] != NULL);

        for (j = 0; j < dimension; ++j)
        {
            arr[i][j] = rand() % ceiling;
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

float** readMatrixFromFile(char* fileName)
{
    float** arr = NULL;

    // First line contains dimension (n x n).
    int dimension = getDimensionFromFile(fileName);

    FILE* fp = fopen(fileName, "r");
    if (fp != NULL)
    {
        // Next line contains matrix contents (space delimited floats).
        arr = (float**)malloc(dimension * sizeof(float*));
        assert(arr != NULL);

        int i, j;
        for (i = 0; i < dimension; ++i)
        {
            arr[i] = (float*)malloc(dimension * sizeof(float));
            assert(arr[i] != NULL);

            for (j = 0; j < dimension; ++j)
            {
                fscanf(fp, "%f", &arr[i][j]);
            }
        }

        fclose(fp);
    }

    return arr;
}

