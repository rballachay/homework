#!/bin/bash

# the first test, run the sequential version
echo "Starting first test, sequential version"
mpicc sequential_quicksort.c -o sequential_quicksort.o
mpirun -np 1 ./sequential_quicksort.o

echo "Starting hypercube test, running with N=2,4,6,8,10,12"
mpicc parallel_quicksort.c -o parallel_quicksort.o
# in the second test
for N in 2 4 # 6 8 10 12
do
    echo "Time for n = $N"
    mpirun -np $N --use-hwthread-cpus ./parallel_quicksort.o 
done


echo "Starting PSRS test, running with N=2,4,6,8,10,12"
mpicc psrs_sorting.c -o psrs_sorting.o
# in the second test
for N in 2 4 # 6 8 10 12
do
    echo "Time for n = $N"
    mpirun -np $N --use-hwthread-cpus ./psrs_sorting.o 
done