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


  int count=3;  //Number of elements each process will receive

  int masterProcess=0; //P0 will populate main array and scatter

  int recvArr[count];   //Used to accumulate scattered elements at each individual process


  const int tag=10;

//  Initialize MPI.

  MPI_Init ( &argc, &argv );  

  startTime = MPI_Wtime ( );

//  Get the number of processes.

  MPI_Comm_size ( MPI_COMM_WORLD, &numProcesses );

//  Get the individual process ID.

  MPI_Comm_rank ( MPI_COMM_WORLD, &id );

  int mainArr[count*numProcesses]; //Static Allocation in this example


//Process 0 populates array and scatters it to others including itself.

  if(id==masterProcess){

     for(int i=0;i<count*numProcesses;i++){
	  mainArr[i]= i;
     }
  }

//PLEASE NOTICE THAT MPI_SCATTER FUNCTION IS OUTSIDE THE IF BLOCK OF MASTER PROCESS 0.
//A COLLECTIVE FUNCTION NEEDS TO BE CALLED BY EVERYONE
//A PARAMETER called 'root' OF THE SCATTER FUNCTION WILL INDICATE WHICH PROCESS IS GOING TO SCATTER ELEMENTS

	/*	MPI_Scatter(
		void* send_data,
		int send_count,
		MPI_Datatype send_datatype,
		void* recv_data,
		int recv_count,
		MPI_Datatype recv_datatype,
		int root,
		MPI_Comm communicator)
	*/

  MPI_Scatter(mainArr,count,MPI_INT,recvArr,count,MPI_INT,masterProcess,MPI_COMM_WORLD);

//Each Process then print their received chunks

  for(int i=0;i<count;i++){
      cout<<"\nP"<<id<<": "<<recvArr[i]<<"\n";	   
  }


//  Terminate MPI.

  elapsedTime = MPI_Wtime ( ) - startTime;

  MPI_Finalize ( );

//  Process 0 prints a termination message.

  if ( id == 0 )
  {
    cout << "\n\nP" << id << ":    Elapsed time = " << elapsedTime << " seconds.\n";	
    cout << "P" << id << ":    Normal end of execution.\n";
  }

  return 0;
}

