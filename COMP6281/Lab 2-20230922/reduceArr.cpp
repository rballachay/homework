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
  int resultArr[count];  //used to accumulate result at final destination

  const int tag=10;

//  Initialize MPI.

  MPI_Init ( &argc, &argv );  

  startTime = MPI_Wtime ( );

//  Get the number of processes.

  MPI_Comm_size ( MPI_COMM_WORLD, &numProcesses );

//  Get the individual process ID.

  MPI_Comm_rank ( MPI_COMM_WORLD, &id );


//Each process populates it's array individually using multiples of their rank in this example
  for(int i=0;i<count;i++){
	arr[i]= id*(i+1);
  }

//Each process does a reduction. The final result will be stored in rank 0 here. You can change to whatever you want.

  MPI_Reduce(&arr,&resultArr,count,MPI_INT,MPI_SUM,finalDestination,MPI_COMM_WORLD);   //SUM

//  MPI_Reduce(&arr,&resultArr,count,MPI_INT,MPI_MAX,finalDestination,MPI_COMM_WORLD);     //MAX


//Print result
  if(id==finalDestination){

	for(int i=0;i<count;i++){
	    cout<<"\nP"<<id<<": "<<resultArr[i]<<"\n";	    
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

