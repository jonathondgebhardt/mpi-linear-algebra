#include <math.h>

/*
 * This library is created from examples shown in class. Minor modifications are
 * made, mostly for coding style to improve readibility.
 *
 * Author: Jonathon Gebhardt
 * Author of original functions: Dr. John Nehrbass
 * Class: CS4900-B90
 */

// Adapted from det_recursive.c.
double findDeterminant(double **a, int start, int end, int dimension)
{
    int i, j, k, m;
    double det = 0;
    double **arr = NULL;

    // Base case.
    if (dimension == 0)
    {
        return 0;
    }

    // Reduction.
    if (dimension == 1)
    {
        det = a[0][0];
    }
    else if (dimension == 2)
    {
        det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    }
    else
    {
        det = 0;

        for (k = start; k < end; k++)
        {
            arr = (double **)malloc((dimension - 1) * sizeof(double *));

            for (i = 0; i < dimension - 1; i++)
            {
                arr[i] = (double *)malloc((dimension - 1) * sizeof(double));
            }

            for (i = 1; i < dimension; i++)
            {
                m = 0;
                for (j = 0; j < dimension; j++)
                {
                    if (j == k)
                    {
                        continue;
                    }

                    arr[i - 1][m] = a[i][j];

                    m++;
                }
            }

            det += pow(-1.0, 1.0 + k + 1.0) * a[0][k] *
                   findDeterminant(arr, 0, dimension - 1, dimension - 1);

            for (i = 0; i < dimension - 1; i++)
            {
                free(arr[i]);
            }

            free(arr);
        }
    }

    return det;
}

// Adapted from vecDot.c.
double dot(double *v1, double *v2, int n)
{
    double sum = 0.0;

    int i;
    for (i = 0; i < n; i++)
    {
        sum += v1[i] * v2[i];
    }

    return sum;
}
