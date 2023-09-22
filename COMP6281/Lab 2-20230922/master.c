
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

int main (int argc, char *argv[])
{
    int numChildren, processId, numParents;

    int sum;

    MPI_Comm childComm;
    
    /* Obtain number of tasks and task ID */
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numParents);

    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    printf ("Parent task id is: %d. Total tasks in parent's world are: %d \n", processId, numParents );

    if ( argc != 2 )
        printf( "usage: %s <number of workers>\n", argv[0] );
    else
        numChildren = atoi( argv[1] ); //Given by user in command line e.g: 'mpirun -np 1 master 4' -> numChildren=4

    /*
    int MPI_Comm_spawn(
                        const char *command, 
                        char *argv[], 
                        int maxprocs,
                        MPI_Info info, 
                        int root, 
                        MPI_Comm comm,
                        MPI_Comm *intercomm, 
                        int array_of_errcodes[])
    */
   
                                                              //MPI_COMM_SELF : Each parent spawns own process
    MPI_Comm_spawn( "c", argv, numChildren, MPI_INFO_NULL, 0, MPI_COMM_SELF, &childComm, MPI_ERRCODES_IGNORE );

    MPI_Reduce(&processId, &sum, 1, MPI_INT, MPI_SUM,MPI_ROOT, childComm);  //Specifies child communicator as comm....USE MPI_ROOT (VERY IMPORTANT)
	 
    MPI_Finalize();

    printf("P %d: SUM: %d\n\n",processId,sum);
    
    return 0;
}
