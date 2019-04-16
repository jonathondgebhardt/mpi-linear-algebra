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

// Attempt to align memory.
typedef struct
{
    double** matrixWithSolution;
    double** matrix;
    double** cofactorMatrix;
    double** inverseMatrix;
    double* xSample;
    double* xSolution;
    double* yMatrix;
    char* fileName;
    double det;
    double error;
    int dimension;
} LinEq;

void showUsage(char*);
LinEq* initLinEq();
void cleanUp(LinEq*);
void print1dMatrix(double*, int);
void print2dMatrix(double**, int, int);
double** createDiagonalMatrix(int);
double** createRandomMatrix(int);
int getDimensionFromFile(char*);
double** getMatrixFromFile(char*);
double* createRandomColumnMatrix(int);
void appendSolutionToMatrix(LinEq*);
void getSolutionFromMatrix(LinEq*);
void rowReduce(double**, int);
void swapRows(double**, int, int, int);
void scalarMultiply(double*, double, int);
double getError(double*, double*, int);
void showResults(LinEq*);

int main(int argc, char* argv[])
{
    int rank, ncpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    int opt, rFlag = 0, dFlag = 0, fFlag = 0, vFlag = 0;
    LinEq* le = initLinEq();

    while((opt = getopt(argc, argv, "r:d:f:v")) != -1)
    {
        switch(opt)
        {
            case 'r':
                rFlag = 1;
                le->dimension = atoi(optarg);
                break;
            case 'd':
                dFlag = 1;
                le->dimension = atoi(optarg);
                break;
            case 'f':
                fFlag = 1;
                le->fileName = optarg;
                break;
            case 'v':
                vFlag = 1;
                break;
            case '?':
                if(rank == 0)
                {
                    fprintf(stderr, "Invalid parameter '%s'\n", opt);

                    showUsage(argv[0]);
                    MPI_Abort(MPI_COMM_WORLD, 1);

                    return 1;
                }
                break;
        }
    }

    if(rank == 0)
    {
        // Validate user input.
        if(rFlag == 0 && dFlag == 0 && fFlag == 0)
        {
            fprintf(stderr,
                    "A value for -r, -d, or -f exclusively is required\n");

            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            cleanUp(le);

            return 1;
        }
        else if(rFlag + dFlag + fFlag > 1)
        {
            fprintf(stderr, "Only one option may be used at a time\n");

            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            cleanUp(le);

            return 1;
        }

        // Seeding rand() isn't necessary for every case, but seed it here once
        // to simplify logic.
        srand(time(NULL));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Populate matrix based on user request.
    if(rank == 0)
    {
        if(rFlag == 1)
        {
            le->matrix = createRandomMatrix(le->dimension);
        }
        else if(dFlag == 1)
        {
            le->matrix = createDiagonalMatrix(le->dimension);
        }
        else
        {
            le->matrix = getMatrixFromFile(le->fileName);
            if(le->matrix == NULL)
            {
                fprintf(stderr, "Error reading from file\n");

                showUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                cleanUp(le);

                return 1;
            }

            le->dimension = getDimensionFromFile(le->fileName);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    // TODO: This part is parallelized.
    // If the determinant is 0, we can't do any meaningful work.
    if((le->det = determinantNehrbass(le->matrix, 0, le->dimension,
                                      le->dimension)) != 0)
    {
        if(rank == 0)
        {
            // TODO: Determine how many tasks are needed to solve for 10. Dish
            // out tasks until done. Gather error.
        }
        else
        {
            // TODO: Worker receive status. If no work, quit.

            le->cofactorMatrix = cofactor(le->matrix, le->dimension);

            le->inverseMatrix =
                transpose(le->matrix, le->cofactorMatrix, le->dimension);

            // Generate a random 'x' to solve Ax = Y.
            le->xSample = createRandomColumnMatrix(le->dimension);

            le->yMatrix = create1dDoubleMatrix(le->dimension);
            int i;
            for(i = 0; i < le->dimension; ++i)
            {
                le->yMatrix[i] = dot(le->matrix[i], le->xSample, le->dimension);
            }

            // Now that we have A, a sample x, and Y, we use A and Y to solve
            // for x using back substitution.
            appendSolutionToMatrix(le);
            rowReduce(le->matrixWithSolution, le->dimension);
            getSolutionFromMatrix(le);

            le->error = getError(le->xSample, le->xSolution, le->dimension);
        }
    }

    double endTime = MPI_Wtime();

    if(rank == 0 && vFlag == 1)
    {
        showResults(le);
    }

    printf("\n-------------------------------------------\n");
    printf("Elapsed time: %lf\n", (endTime - startTime));
    printf("Error: %lf", le->error);
    printf("\n-------------------------------------------\n\n");

    cleanUp(le);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}

void showUsage(char* applicationName)
{
    printf("Usage: %s [-r m] [-d n] [-f fileName] [-v]\n", applicationName);
    printf("\t-r: m x m matrix filled with random numbers\n");
    printf(
        "\t-d: n x n diagonal matrix where the values are the row numbers\n");
    printf(
        "\t-f: read a matrix from a file where the first line is the square "
        "dimension and the second line contains space delimited elements\n");
    printf("\t-v: verbose output\n");
}

LinEq* initLinEq()
{
    LinEq* le = malloc(sizeof(LinEq));
    le->matrixWithSolution = NULL;
    le->matrix = NULL;
    le->cofactorMatrix = NULL;
    le->inverseMatrix = NULL;
    le->xSample = NULL;
    le->xSolution = NULL;
    le->yMatrix = NULL;
    le->det = -1;
    le->error = -1;
    le->dimension = -1;

    return le;
}

void cleanUp(LinEq* le)
{
    if(le != NULL)
    {
        int i;

        if(le->matrix != NULL)
        {
            for(i = 0; i < le->dimension; ++i)
            {
                free(le->matrix[i]);
            }

            free(le->matrix);
        }

        if(le->cofactorMatrix != NULL)
        {
            for(i = 0; i < le->dimension; ++i)
            {
                free(le->cofactorMatrix[i]);
            }

            free(le->cofactorMatrix);
        }

        if(le->inverseMatrix != NULL)
        {
            for(i = 0; i < le->dimension; ++i)
            {
                free(le->inverseMatrix[i]);
            }

            free(le->inverseMatrix);
        }

        if(le->xSolution != NULL)
        {
            free(le->xSolution);
        }

        if(le->xSample != NULL)
        {
            free(le->xSample);
        }

        if(le->yMatrix != NULL)
        {
            free(le->yMatrix);
        }

        free(le);
    }
}

void print1dMatrix(double* arr, int dimension)
{
    int i;
    for(i = 0; i < dimension; ++i)
    {
        printf("%9.4f ", arr[i]);
    }

    printf("\n");
}

void print2dMatrix(double** arr, int row, int col)
{
    int i;
    for(i = 0; i < row; ++i)
    {
        int j;
        for(j = 0; j < col; ++j)
        {
            printf("%9.4f ", arr[i][j]);
        }

        printf("\n");
    }
}

double** createDiagonalMatrix(int dimension)
{
    double** arr = create2dDoubleMatrix(dimension);

    int i, j;
    for(i = 0; i < dimension; ++i)
    {
        for(j = 0; j < dimension; ++j)
        {
            if(i == j)
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
    for(i = 0; i < dimension; ++i)
    {
        for(j = 0; j < dimension; ++j)
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
    if(fp != NULL)
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
    if(fp != NULL)
    {
        // First line contains dimension (n x n).
        int dimension;
        fscanf(fp, "%d", &dimension);
        assert(dimension != 0);

        // Next line contains matrix contents (space delimited doubles).
        arr = create2dDoubleMatrix(dimension);

        int i, j;
        for(i = 0; i < dimension; ++i)
        {
            for(j = 0; j < dimension; ++j)
            {
                fscanf(fp, "%lf", &arr[i][j]);
            }
        }

        fclose(fp);
    }

    return arr;
}

double* createRandomColumnMatrix(int dimension)
{
    double* arr = create1dDoubleMatrix(dimension);

    int i, ceiling = 100;
    for(i = 0; i < dimension; ++i)
    {
        arr[i] = (double)rand() / ((double)RAND_MAX + 1) * ceiling;
    }

    return arr;
}

void getSolutionFromMatrix(LinEq* le)
{
    le->xSolution = create1dDoubleMatrix(le->dimension);
    int i;
    for(i = 0; i < le->dimension; ++i)
    {
        le->xSolution[i] = le->matrixWithSolution[i][le->dimension];
    }
}

void appendSolutionToMatrix(LinEq* le)
{
    if(le != NULL && le->matrix != NULL && le->yMatrix != NULL)
    {
        le->matrixWithSolution =
            (double**)malloc(le->dimension * sizeof(double*));
        int i;
        for(i = 0; i < le->dimension; ++i)
        {
            le->matrixWithSolution[i] =
                (double*)malloc((le->dimension + 1) * sizeof(double));
        }

        for(i = 0; i < le->dimension; ++i)
        {
            int j;
            for(j = 0; j < le->dimension; ++j)
            {
                le->matrixWithSolution[i][j] = le->matrix[i][j];
            }

            le->matrixWithSolution[i][le->dimension] = le->yMatrix[i];
        }
    }
}

// Pseudo-code taken from
// https://www.rosettacode.org/wiki/Reduced_row_echelon_form.
// I should mention that there are implementations for several languages on this
// page. I did not directly copy from these implementations but implemented my
// own version based on the pseudo code and used the reference implementations
// as a guideline.
void rowReduce(double** arr, int dimension)
{
    int r, lead = 0, rowCount = dimension, columnCount = dimension + 1;

    for(r = 0; r < rowCount; ++r)
    {
        if(columnCount <= lead)
        {
            return;
        }

        int i = r;

        while(arr[i][lead] == 0)
        {
            ++i;

            if(rowCount == i)
            {
                i = r;

                ++lead;

                if(columnCount == lead)
                {
                    return;
                }
            }
        }

        swapRows(arr, i, r, columnCount);

        if(arr[r][lead] != 0)
        {
            scalarMultiply(arr[r], (1 / arr[r][lead]), columnCount);
        }

        for(i = 0; i < rowCount; ++i)
        {
            if(i != r)
            {
                int j;
                double leadValue = -arr[i][lead];
                for(j = 0; j < columnCount; ++j)
                {
                    arr[i][j] += leadValue * arr[r][j];
                }
            }
        }

        ++lead;
    }
}

void swapRows(double** arr, int first, int second, int dimension)
{
    if(arr != NULL && arr[first] != NULL && arr[second] != NULL)
    {
        int i;
        for(i = 0; i < dimension; ++i)
        {
            double copy = arr[first][i];
            arr[first][i] = arr[second][i];
            arr[second][i] = copy;
        }
    }
}

void scalarMultiply(double* arr, double scalar, int dimension)
{
    if(arr != NULL)
    {
        int i;
        for(i = 0; i < dimension; ++i)
        {
            arr[i] *= scalar;
        }
    }
}

double getError(double* a, double* b, int dimension)
{
    int i;
    double sumOfSquaredDiff = 0.0;
    for(i = 0; i < dimension; ++i)
    {
        sumOfSquaredDiff += pow(fabs(a[i] - b[i]), 2);
    }

    return sqrt(sumOfSquaredDiff) / (double)dimension;
}

void showResults(LinEq* le)
{
    if(le != NULL && le->dimension != -1)
    {
        if(le->matrix != NULL)
        {
            printf("\nThe given matrix:\n");
            print2dMatrix(le->matrix, le->dimension, le->dimension);
        }

        if(le->det != -1)
        {
            printf("\nDeterminant of the given matrix:\n  %.4lf\n", le->det);
        }

        if(le->inverseMatrix != NULL)
        {
            printf("\nThe inverse of the given matrix:\n");
            print2dMatrix(le->inverseMatrix, le->dimension, le->dimension);
        }

        if(le->xSample != NULL)
        {
            printf("\nThe generated x sample:\n");
            print1dMatrix(le->xSample, le->dimension);
        }

        if(le->yMatrix != NULL)
        {
            printf("\nThe computed y matrix:\n");
            print1dMatrix(le->yMatrix, le->dimension);
        }

        if(le->xSolution != NULL)
        {
            printf("\nThe computed solution for Ax = Y:\n");
            print1dMatrix(le->xSolution, le->dimension);
        }

        if(le->error != -1)
        {
            printf("\nThe computed error:\n  %.4lf\n", le->error);
        }
    }
}
