# mpi-linear-algebra
A program written in C that leverages Slurm and MPI to to solve the linear equation Ax = Y, where A and Y are square matrices and x is a column matrix. 

- Class: CS4900-B90 -- HPC & Parallel Programming
- Semester: Spring 2019
- Instructor: Dr. John Nehrbass

## Usage: ./hw03 [-r m] [-d n] [-f fileName] [-v]
* -r: m x m matrix filled with random numbers
* -d: n x n diagonal matrix where the values are the row numbers
* -f: read a matrix from a file where the first line is the square dimension and the second line contains space delimited elements
* -v: verbose output

## Input
Use one of the input parameters based on the above usage.

## Output
Error and execution time are reported at a minimum. If the verbose option is used, various stages of the process are reported as well (e.g., the input matrix).

## Requirements
- gcc v4.8.5
- Slurm v17.11.5
- MPI v2.1.0

You may be able to execute this application with lesser versions but these are the versions of requirements the application was developed on.

## Compilation
- ./compile.sh hw03 [nodes] [tasks per node]
- Add appropriate command line arguments to last line in approximate.batch
