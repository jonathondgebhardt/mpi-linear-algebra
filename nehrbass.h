#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * This library is created from examples shown in class. Minor modifications are
 * made, mostly for coding style to improve readibility.
 *
 * Author: Jonathon Gebhardt
 * Author of original functions: Dr. John Nehrbass
 * Class: CS4900-B90
 */

double* create1dDoubleMatrix(int dimension)
{
    double* arr = (double*)malloc(dimension * sizeof(double));
    assert(arr != NULL);

    return arr;
}

double** create2dDoubleMatrix(int dimension)
{
    double** arr = (double**)malloc(dimension * sizeof(double*));
    assert(arr != NULL);

    int i;
    for (i = 0; i < dimension; ++i)
    {
        arr[i] = (double*)malloc(dimension * sizeof(double));
        assert(arr[i] != NULL);
    }

    return arr;
}

// Adapted from det_recursive.c.
double determinantNehrbass(double** a, int start, int end, int dimension)
{
    int i, j, k, m;
    double det = 0;
    double** arr = NULL;

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
            arr = create2dDoubleMatrix(dimension - 1);

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
                   determinantNehrbass(arr, 0, dimension - 1, dimension - 1);

            for (i = 0; i < dimension - 1; i++)
            {
                free(arr[i]);
            }
            free(arr);
        }
    }

    return det;
}

// Adapted from scalMatInv.c which was taken from
// http://www.ccodechamp.com/c-program-to-find-inverse-of-matrix/
double determinantChamp(double** arr, float dimension)
{
    if (dimension == 1)
    {
        return arr[0][0];
    }

    float sign = 1, det = 0;
    double** b = create2dDoubleMatrix(dimension);

    int i;
    for (i = 0; i < dimension; i++)
    {
        int j, m = 0, n = 0;
        for (j = 0; j < dimension; j++)
        {
            int k;
            for (k = 0; k < dimension; k++)
            {
                b[j][k] = 0;

                if (j != 0 && k != i)
                {
                    b[m][n] = arr[j][k];

                    if (n < (dimension - 2))
                    {
                        n++;
                    }
                    else
                    {
                        n = 0;
                        m++;
                    }
                }
            }
        }

        det = det + sign * (arr[0][i] * determinantChamp(b, dimension - 1));
        sign = -1 * sign;
    }

    return det;
}

// Adapted from vecDot.c.
double dot(double* v1, double* v2, int n)
{
    double sum = 0.0;

    int i;
    for (i = 0; i < n; i++)
    {
        sum += v1[i] * v2[i];
    }

    return sum;
}

// Adapted from scalMatInv.c which was taken from
// http://www.ccodechamp.com/c-program-to-find-inverse-of-matrix/
double** cofactor(double** arr, int dimension)
{
    double** cofactorMatrix = create2dDoubleMatrix(dimension);
    double** minorsMatrix = create2dDoubleMatrix(dimension);

    int i;
    for (i = 0; i < dimension; i++)
    {
        int j;
        for (j = 0; j < dimension; j++)
        {
            int k, p = 0, q = 0;
            for (k = 0; k < dimension; k++)
            {
                int m;
                for (m = 0; m < dimension; m++)
                {
                    if (k != i && m != j)
                    {
                        minorsMatrix[p][q] = arr[k][m];

                        if (q < (dimension - 2))
                        {
                            q++;
                        }
                        else
                        {
                            q = 0;
                            p++;
                        }
                    }
                }
            }

            cofactorMatrix[i][j] =
                pow(-1, i + j) * determinantChamp(minorsMatrix, dimension - 1);
        }
    }

    for (i = 0; i < dimension; ++i)
    {
        free(minorsMatrix[i]);
    }
    free(minorsMatrix);

    return cofactorMatrix;
}

// Adapted from scalMatInv.c which was taken from
// http://www.ccodechamp.com/c-program-to-find-inverse-of-matrix/
double** transpose(double** arr, double** fac, int dimension)
{
    double b[25][25], det;
    double** inverse = create2dDoubleMatrix(dimension);

    int i, j;
    for (i = 0; i < dimension; i++)
    {
        for (j = 0; j < dimension; j++)
        {
            b[i][j] = fac[j][i];
        }
    }

    det = determinantChamp(arr, dimension);

    for (i = 0; i < dimension; i++)
    {
        for (j = 0; j < dimension; j++)
        {
            inverse[i][j] = b[i][j] / det;
        }
    }

    return inverse;
}
