/**********************************************************************
 * FILE: mpi_pi_reduce.c
 * OTHER FILES: dboard.c
 * DESCRIPTION:  
 *   MPI pi Calculation Example - C Version 
 *   Collective Communication example:  
 *   This program calculates pi using a "dartboard" algorithm.  See
 *   Fox et al.(1988) Solving Problems on Concurrent Processors, vol.1
 *   page 207.  All processes contribute to the calculation, with the
 *   master averaging the values for pi. This version uses mpc_reduce to 
 *   collect results
 * AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
 *   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95) 
 * LAST REVISED: 06/13/13 Blaise Barney
**********************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double dboard(int darts);

#define DARTS 50000     /* number of throws at dartboard */
#define ROUNDS 100      /* number of times "darts" is iterated */
#define NUM_WORKERS 1   /* Number of worker processes to spawn */

int main(int argc, char *argv[]) {
    int rank, num_procs, provided;
    MPI_Status status;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); // Initialize MPI with thread support

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // The master process will handle dynamic process spawning
    if (rank == 0) {
        double avepi = 0;
        int i;

        for (i = 0; i < ROUNDS; i++) {
            double homepi = dboard(DARTS);

            // Perform dynamic process spawning for the specified number of worker processes
            char worker_executable[256]; // Adjust the size as needed
            sprintf(worker_executable, "./%s", argv[0]); // Use the same executable
            char* worker_argv[2] = {worker_executable, NULL};

            MPI_Comm child_comm;
            MPI_Comm_spawn(worker_executable, worker_argv, NUM_WORKERS, MPI_INFO_NULL, 0, MPI_COMM_SELF, &child_comm, MPI_ERRCODES_IGNORE);

            // Collect results from worker processes
            double workerpi = 0;
            for (int j = 0; j < NUM_WORKERS; j++) {
                double temp;
                printf("%.3f", temp);
                MPI_Recv(&temp, 1, MPI_DOUBLE, j, 0, child_comm, &status);
                workerpi += temp;
            }

            // Finalize the worker processes
            MPI_Comm_disconnect(&child_comm);

            // Compute the average pi value
            avepi = ((avepi * i) + homepi + workerpi) / (i + 2);
            printf("After %8d throws, average value of pi = %10.8f\n", (DARTS * (i + 2)), avepi);
        }

        printf("\nReal value of PI: 3.1415926535897 \n");
    }
    // Worker processes execute simulations and send results to the master
    else {
        double workerpi = dboard(DARTS);
        MPI_Send(&workerpi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_SELF);
    }

    MPI_Finalize();
    return 0;
}



/**************************************************************************
* subroutine dboard
* DESCRIPTION:
*   Used in pi calculation example codes. 
*   See mpi_pi_send.c and mpi_pi_reduce.c  
*   Throw darts at board.  Done by generating random numbers 
*   between 0 and 1 and converting them to values for x and y 
*   coordinates and then testing to see if they "land" in 
*   the circle."  If so, score is incremented.  After throwing the 
*   specified number of darts, pi is calculated.  The computed value 
*   of pi is returned as the value of this function, dboard. 
*
*   Explanation of constants and variables used in this function:
*   darts       = number of throws at dartboard
*   score       = number of darts that hit circle
*   n           = index variable
*   r           = random number scaled between 0 and 1
*   x_coord     = x coordinate, between -1 and 1
*   x_sqr       = square of x coordinate
*   y_coord     = y coordinate, between -1 and 1
*   y_sqr       = square of y coordinate
*   pi          = computed value of pi
****************************************************************************/

double dboard(int darts)
{
#define sqr(x)	((x)*(x))
long random(void);
double x_coord, y_coord, pi, r; 
int score, n;
unsigned int cconst;  /* must be 4-bytes in size */
/*************************************************************************
 * The cconst variable must be 4 bytes. We check this and bail if it is
 * not the right size
 ************************************************************************/
if (sizeof(cconst) != 4) {
   printf("Wrong data size for cconst variable in dboard routine!\n");
   printf("See comments in source file. Quitting.\n");
   exit(1);
   }
   /* 2 bit shifted to MAX_RAND later used to scale random number between 0 and 1 */
   cconst = 2 << (31 - 1);
   score = 0;

   /* "throw darts at board" */
   for (n = 1; n <= darts; n++)  {
      /* generate random numbers for x and y coordinates */
      r = (double)random()/cconst;
      x_coord = (2.0 * r) - 1.0;
      r = (double)random()/cconst;
      y_coord = (2.0 * r) - 1.0;

      /* if dart lands in circle, increment score */
      if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)
           score++;
      }

/* calculate pi */
pi = 4.0 * (double)score/(double)darts;
return(pi);
} 


