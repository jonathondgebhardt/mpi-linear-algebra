#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void showUsage(char*);
int** createDiagonalMatrix(int dimension);

int main(int argc, char* argv[])
{
    int rank, ncpu;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    int opt, rFlag = 0, dFlag = 0, dimension;

    while((opt = getopt(argc, argv, "r:d:")) != -1)
    {
        // Both args are used as dimensions
        switch(opt)
        {
            case 'r': // Random numbers
                rFlag = 1;
                dimension = atoi(optarg);
                break;
            case 'd': // Diagonal matrix
                dFlag = 1;
                dimension = atoi(optarg);
                break;
            case '?':
                if(rank == 0)
                {
                    showUsage(argv[0]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
        }
    }

    if(rank == 0)
    {
        if(rFlag == 0 && dFlag == 0)
        {
            fprintf(stderr, "A value for -r or -d is required\n");
            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        else if(rFlag == 1 && dFlag == 1)
        {
            fprintf(stderr, "Specify a value for EITHER -r OR -d\n");
            showUsage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    int **matrix;
    if(rFlag == 1)
    {
        // Create random nxn matrix
    }
    else
    {
        // Create diagonal matrix
        matrix = createDiagonalMatrix(dimension);

        int i;
        for(i = 0; i < dimension; ++i)
        {
            int j;
            for(j = 0; j < dimension; ++j)
            {
                printf("%d  ", matrix[i][j]);
            }

            printf("\n");
        }
    }

    int i;
    for(i = 0; i < dimension; ++i)
    {
        free(matrix[i]);
    }

    free(matrix);
    
    return 0;
}

void showUsage(char* applicationName)
{
    printf("Usage: %s [-r m] [-d n]\n", applicationName);
    printf("\t-r: m x m matrix filled with random numbers\n");
    printf("\t-d: n x n diagonal matrix where the values are the row numbers\n");
}

int** createDiagonalMatrix(int dimension)
{
    int** arr = (int**) malloc(dimension * sizeof(int*));

    int i, j;
    for(i = 0; i < dimension; ++i)
    {
        arr[i] = (int*) malloc(dimension * sizeof(int));

        for(j = 0; j < dimension; ++j)
        {
            if(i == j)
            {
                arr[i][j] = j+1;
            }
            else
            {
                arr[i][j] = 0;
            }
        }
    }

    return arr;
}
