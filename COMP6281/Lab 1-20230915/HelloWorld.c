/*The Parallel Hello World Program*/
#include <stdio.h>
#include <mpi.h>

main(int argc, char **argv)
{
   int node;  //Process Id (also called Rank in MPI) Rank 0 is the master process.
   
   MPI_Init(&argc,&argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &node);
     
   printf("Hello World from Node %d\n",node);
            
   MPI_Finalize();
}
