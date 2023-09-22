
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int main (int argc, char *argv[])
{
    int	processId,	        /* task ID - also used as seed number */
        numChildren;       /* number of tasks */


    int sum;

    /* Obtain number of tasks and task ID */

    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numChildren);

    MPI_Comm_rank( MPI_COMM_WORLD, &processId );

    MPI_Comm commParent;  //Parent Communicator

    MPI_Comm_get_parent(&commParent);

    MPI_Reduce(&processId, &sum, 1, MPI_INT, MPI_SUM,0, commParent);   //SUM OF PROCESS IDs...CAN DO OTHER OPERATIONS AS WELL
	
    printf ("Child task %d has started with process id: %d \n", processId, getpid());

	
    MPI_Finalize();

    return 0;
}
