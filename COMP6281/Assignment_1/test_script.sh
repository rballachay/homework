#!/bin/bash

# the first test, run the sequential version
echo "Starting first test, serial version"
mpicc mpi_pi_reduce_serial.c -o pi_program_serial.o
mpirun -np 1 ./pi_program_serial.o 

echo "Starting second test, running with N=2,4,6,8,10,12"
mpicc pisim.c -o pisim.o -std=c11 -lm
mpicc master.c -o pi_program.o -std=c11 -lm
# in the second test
for N in 2 4 6 8 10 12
do
    echo "Time for n = $N"
    mpirun -np 1 --use-hwthread-cpus ./pi_program.o $N
done
