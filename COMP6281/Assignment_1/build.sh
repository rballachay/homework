#!/bin/bash

mpicc pisim.c -o pisim.o
mpicc master.c -o master.o
mpiexec -np 1 --use-hwthread-cpus master.o 2