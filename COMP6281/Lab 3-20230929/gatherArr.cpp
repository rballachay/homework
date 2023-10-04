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


  int count=3;  //Number of elements in each array
  int arr[count]; //Static allocation in this example

  int finalDestination=0;

  const int tag=10;

  int gatherArr[count*numProcesses];  //used to accumulate result at final destination

//  Initialize MPI.

  MPI_Init ( &argc, &argv );  

  startTime = MPI_Wtime ( );

//  Get the number of processes.

  MPI_Comm_size ( MPI_COMM_WORLD, &numProcesses );

//  Get the individual process ID.

  MPI_Comm_rank ( MPI_COMM_WORLD, &id );

//Each process populates it's array individually
  for(int i=0;i<count;i++){
	arr[i]= id;
  }

//Each process does a gather. The final result will be stored in rank 0 here.

	/*	MPI_Gather(
		void* send_data,
		int send_count,
		MPI_Datatype send_datatype,
		void* recv_data,
		int recv_count,
		MPI_Datatype recv_datatype,
		int root,
		MPI_Comm communicator)
	*/

//For recvData, only root process needs it. Others can pass NULL as param.

 MPI_Gather(arr,count,MPI_INT,gatherArr,count,MPI_INT,finalDestination,MPI_COMM_WORLD);

// ALTERNATE SYNTAX:

/*
if(id==finalDestination){
 MPI_Gather(&arr,count,MPI_INT,gatherArr,count,MPI_INT,finalDestination,MPI_COMM_WORLD);
}

else{
 MPI_Gather(&arr,count,MPI_INT,NULL,0,MPI_INT,finalDestination,MPI_COMM_WORLD);
}

*/


//Print result
  if(id==finalDestination){

	for(int i=0;i<count*numProcesses;i++){
	    cout<<"\nP"<<id<<": "<<gatherArr[i]<<"\n";	    
	}
  cout<<"\n";

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

