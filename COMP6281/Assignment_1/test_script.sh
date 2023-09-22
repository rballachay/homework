#!/bin/bash

# the first test, run the sequential version
echo "Starting first test, serial version"
time mpirun -np 1 ./pi_program_serial.o  > /dev/null  2>&1

echo "Starting second test, running with N=2,4,6,8,10"
# in the second test
for N in 2 4 6 8
do
    echo "Time for n = $N"
    time mpirun -np $N ./pi_program.o > /dev/null  2>&1
done

