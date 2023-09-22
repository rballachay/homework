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

//  Initialize MPI.

  MPI_Init ( &argc, &argv );  

  startTime = MPI_Wtime ( );

//  Get the number of processes.

  MPI_Comm_size ( MPI_COMM_WORLD, &numProcesses );

//  Get the individual process ID.

  MPI_Comm_rank ( MPI_COMM_WORLD, &id );

//  Process 0 prints an introductory message.

  if ( id == 0 ) 
  {
    cout << "\nP" << id << ":  The number of processes is " << numProcesses << "\n\n";
  }

//  Every process prints a hello.

  cout << "P" << id << ": Hello World\n";


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

