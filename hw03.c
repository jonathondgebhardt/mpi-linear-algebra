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
void print1dMatrix(double*, int);
void print2dMatrix(double**, int, int);
double** createDiagonalMatrix(int);
double** createRandomMatrix(int);
int getDimensionFromFile(char*);
double** getMatrixFromFile(char*);
double* createRandomColumnMatrix(int);
void rowReduce(double**, int);
void swapRows(double**, int, int, int);
void scalarMultiply(double*, double, int);
double getError(double*, double*, int);

int main(int argc, char* argv[])
{
    int rank, ncpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    double** matrix;
    char* fileName;
    int dimension;

    // Seeding rand() isn't necessary for every case, but seed it here once
    // to simplify logic.
    srand(time(NULL));

    // Initialize environment.
    if(rank == 0)
    {
        int opt, rFlag = 0, dFlag = 0, fFlag = 0;
        while((opt = getopt(argc, argv, "r:d:f:")) != -1)
        {
            switch(opt)
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

        // Validate user input.
        if(rFlag == 0 && dFlag == 0 && fFlag == 0)
        {
            fprintf(stderr,
                    "A value for -r, -d, or -f exclusively is required\n");

            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);

            return 1;
        }
        else if(rFlag + dFlag + fFlag > 1)
        {
            fprintf(stderr, "Only one option may be used at a time\n");

            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);

            return 1;
        }

        // Instantiate matrix based on user's request.
        if(rFlag == 1)
        {
            matrix = createRandomMatrix(dimension);
        }
        else if(dFlag == 1)
        {
            matrix = createDiagonalMatrix(dimension);
        }
        else
        {
            matrix = getMatrixFromFile(fileName);
            if(matrix == NULL)
            {
                fprintf(stderr, "Error reading from file\n");

                showUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);

                free(matrix[0]);
                free(matrix);

                return 1;
            }

            dimension = getDimensionFromFile(fileName);
        }

        // If the determinant is zero, we can't do any meaningful work.
        double det;
        if((det = determinantNehrbass(matrix, 0, dimension, dimension)) == 0)
        {
            fprintf(stderr, "Determinant of the given matrix is 0\n");

            MPI_Abort(MPI_COMM_WORLD, 1);

            free(matrix[0]);
            free(matrix);

            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // If the determinant is 0, we can't do any meaningful work.
    int stopTask = 0, startTask = 1;
    if(rank == 0)
    {
        // Front load passing the given array.
        int workersWithTask[ncpu - 1];
        int i;
        for(i = 1; i < ncpu; ++i)
        {
            workersWithTask[i - 1] = 0;
            MPI_Send(&dimension, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            // https://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi
            MPI_Send(&(matrix[0][0]), dimension * dimension, MPI_DOUBLE, i, 1,
                     MPI_COMM_WORLD);
        }
    }

    double cumulativeError = 0.0;
    int numTasks = 10;

    printf("[%d] Beginning work\n", rank);
    double startTime = MPI_Wtime();

    // Below is my attempt at implementing a worker-pool-like task tracker.
    /*
            do
            {
                for (i = 1; i < ncpu; ++i)
                {
                    if(numTasks == 0)
                    {
                        break;
                    }

                    // Send an instruction to a worker.
                    MPI_Send(&startTask, 1, MPI_INT, i, 1, MPI_COMM_WORLD);

                    workersWithTask[i-1] = 1;

                   --numTasks;
                }

                for(i = 1; i < ncpu; ++i)
                {
                    if(workersWithTask[i-1] == 1)
                    {
                        double error = 0.0;
                        MPI_Recv(&error, 1, MPI_DOUBLE, i, 1,
       MPI_COMM_WORLD, MPI_STATUS_IGNORE); cumulativeError += error;

                        workersWithTask[i-1] = 0;
                    }
                }

                if(numTasks < 0)
                {
                    MPI_Abort(MPI_COMM_WORLD, 1);

                    return 1;
                }

            } while(numTasks != 0);
    */

    int currentNode = 1;
    for(i = 0; i < numTasks; ++i)
    {
        double error = 0.0;
        // Send an instruction to a worker.
        MPI_Send(&startTask, 1, MPI_INT, currentNode, 1, MPI_COMM_WORLD);

        MPI_Recv(&error, 1, MPI_DOUBLE, currentNode, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        cumulativeError += error;

        currentNode = (currentNode + 1) % ncpu;

        // Don't assign work to self.
        if(currentNode == 0)
        {
            currentNode = 1;
        }
    }

    double endTime = MPI_Wtime();
    printf("[%d] Work concluded, stopping workers\n", rank);

    for(i = 1; i < ncpu; ++i)
    {
        MPI_Send(&stopTask, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
    }

    printf("\n-------------------------------------------\n");
    printf("Elapsed time: %lf\n", (endTime - startTime));
    printf("Error: %lf", cumulativeError);
    printf("\n-------------------------------------------\n\n");

    free(matrix[0]);
    free(matrix);
}
else
{
    MPI_Recv(&dimension, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // https://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi
    matrix = create2dDoubleMatrix(dimension, dimension);
    MPI_Recv(&(matrix[0][0]), dimension * dimension, MPI_DOUBLE, 0, 1,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Continue work until task master says to stop.
    while(1)
    {
        int msg;
        MPI_Recv(&msg, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(msg == stopTask)
        {
            break;
        }

        printf("[%d] Starting task\n", rank);

        // Generate a random 'x' to solve Ax = Y.
        double* xSample = createRandomColumnMatrix(dimension);

        double* yMatrix = create1dDoubleMatrix(dimension);
        int i;
        for(i = 0; i < dimension; ++i)
        {
            yMatrix[i] = dot(matrix[i], xSample, dimension);
        }

        double** matrixWithSolution =
            create2dDoubleMatrix(dimension, dimension + 1);
        for(i = 0; i < dimension; ++i)
        {
            int j;
            for(j = 0; j < dimension; ++j)
            {
                matrixWithSolution[i][j] = matrix[i][j];
            }

            matrixWithSolution[i][dimension] = yMatrix[i];
        }

        free(yMatrix);

        // Now that we have A, a sample x, and Y, we use A and Y to solve
        // for x using back substitution.
        rowReduce(matrixWithSolution, dimension);

        double* xSolution = create1dDoubleMatrix(dimension);
        for(i = 0; i < dimension; ++i)
        {
            xSolution[i] = matrixWithSolution[i][dimension];
        }

        double error = getError(xSample, xSolution, dimension);
        MPI_Send(&error, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        free(xSample);

        free(matrixWithSolution[0]);
        free(matrixWithSolution);

        free(xSolution);
    }
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
    double** arr = create2dDoubleMatrix(dimension, dimension);

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
    double** arr = create2dDoubleMatrix(dimension, dimension);

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
        arr = create2dDoubleMatrix(dimension, dimension);

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

    printf("Returning contents from file\n");

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
        if(lead >= columnCount)
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
