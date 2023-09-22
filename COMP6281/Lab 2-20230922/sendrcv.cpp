# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>
# include <mpi.h>

using namespace std;

int main ( int argc, char *argv[] ) {

  int id;  //process Id  (Rank in MPI)

  int numProcesses;

  double startTime, elapsedTime;

  int sendVal;
  int recvVal;

  int source=0;
  int destination=1;

  const int tag=10;

//  Initialize MPI.

  MPI_Init ( &argc, &argv );  

  startTime = MPI_Wtime ( );

//  Get the number of processes.

  MPI_Comm_size ( MPI_COMM_WORLD, &numProcesses );

//  Get the individual process ID.

  MPI_Comm_rank ( MPI_COMM_WORLD, &id );


  if ( numProcesses<2 ) 
  {
    cout << "Only 1 process...Cannot proceed. Terminating Program";
    
  }

  else if ( id == 0 ) 
  {
    sendVal=555;

    MPI_Send(&sendVal,1,MPI_INT,destination,tag,MPI_COMM_WORLD);

  }


  else if ( id == 1 ) 
  {

    MPI_Recv(&recvVal,1,MPI_INT,source,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    cout << "\nP" << id << ":  Value received: " << recvVal << "\n\n";
  }

  else{

   cout << "\nP" << id << ":  Nothing to do :P\n";  

  }



//  Terminate MPI.

  elapsedTime = MPI_Wtime ( ) - startTime;

  MPI_Finalize ( );

//  Process 0 prints a termination message.

  if ( id == 0 )
  {
    cout << "\nP" << id << ":    Elapsed time = " << elapsedTime << " seconds.\n";	
    cout << "P" << id << ":    Normal end of execution.\n";
  }

  return 0;
}

