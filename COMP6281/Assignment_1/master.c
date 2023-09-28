
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define DARTS 50000     /* number of throws at dartboard */
#define ROUNDS 10000      /* number of times "darts" is iterated */
#define MASTER 0        /* task ID of master task */

int main (int argc, char *argv[])
{
    char ** newargv = malloc(sizeof(char *)*4);//create new argv for childs

    int numChildren, processId, numParents;
    double startTime, elapsedTime, sum, pi;

    MPI_Comm childComm;
    
    /* Obtain number of tasks and task ID */
    MPI_Init(&argc,&argv);

    startTime = MPI_Wtime ( );

    MPI_Comm_size(MPI_COMM_WORLD, &numParents);

    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    printf ("Parent task id is: %d. Total tasks in parent's world are: %d \n", processId, numParents );

    if ( argc != 2 )
        printf( "usage: %s <number of workers>\n", argv[0] );
    else
        numChildren = atoi( argv[1] ); //Given by user in command line e.g: 'mpirun -np 1 master 4' -> numChildren=4

    // Casting the integer to a double
    double doubleRounds;
    doubleRounds = (double)ROUNDS;
    double roundsPerChild = ceil(doubleRounds / numChildren); // truncating integer division

    char rounds[50];
    char darts[50]; // Assuming a buffer large enough to hold the string

    // Use sprintf to convert the integer to a string
    sprintf(darts, "%d", DARTS);
    sprintf(rounds, "%f", roundsPerChild);

   // these are the arguments we are going to pass to our child process
    char* spawn_argv[] = {rounds, darts, NULL};
                                                              //MPI_COMM_SELF : Each parent spawns own process
    MPI_Comm_spawn( "./pisim.o", spawn_argv, numChildren, MPI_INFO_NULL, 0, MPI_COMM_SELF, &childComm, MPI_ERRCODES_IGNORE );

    MPI_Reduce(&pi, &sum, 1, MPI_DOUBLE, MPI_SUM,MPI_ROOT, childComm);  //Specifies child communicator as comm....USE MPI_ROOT (VERY IMPORTANT)
	 
    elapsedTime = MPI_Wtime ( ) - startTime;
    MPI_Finalize();

    printf("Average pi: %.7f\n\n",sum/(roundsPerChild*numChildren));

    if ( processId == 0 )
    {
     printf("Elapsed time =  %f seconds.\n",elapsedTime);	
    }
    
    return 0;
}
