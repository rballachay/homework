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

  int partialSum=0; //Each process does a partialSum and sends the data back to master using gather

  int totalSum=0; //FINAL SUM THAT WILL BE CALCULATED BY MASTER ONLY
 
 const int tag=10;

//  Initialize MPI.

  MPI_Init ( &argc, &argv );  

  startTime = MPI_Wtime ( );

//  Get the number of processes.

  MPI_Comm_size ( MPI_COMM_WORLD, &numProcesses );

//  Get the individual process ID.

  MPI_Comm_rank ( MPI_COMM_WORLD, &id );

  int mainArr[count*numProcesses]; //Static Allocation in this example

  int partialSumArr[numProcesses];  //Used to accumulate Gathered partial sums at master process


//Process 0 populates array and scatters it to others including itself.
  if(id==masterProcess){

     for(int i=0;i<count*numProcesses;i++){
	  mainArr[i]= i;

     }
  }

  MPI_Scatter(mainArr,count,MPI_INT,recvArr,count,MPI_INT,masterProcess,MPI_COMM_WORLD);

//Each Process then prints their received chunks and do a partial sum

  for(int i=0;i<count;i++){

      partialSum+=recvArr[i];

  //    cout<<"\nP"<<id<<": "<<recvArr[i]<<"\n";	   
  }


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

 MPI_Gather(&partialSum,1,MPI_INT,partialSumArr,1,MPI_INT,masterProcess,MPI_COMM_WORLD);


// ALTERNATE SYNTAX:

/*
if(id==masterProcess){
 MPI_Gather(&partialSum,1,MPI_INT,partialSumArr,1,MPI_INT,masterProcess,MPI_COMM_WORLD);
}

else{
 MPI_Gather(&partialSum,1,MPI_INT,NULL,0,MPI_INT,masterProcess,MPI_COMM_WORLD);
}

*/



  //NOW MASTER PROCESS (PO) DOES THE FINAL SUM AND PRINTS THE RESULTS

  if(id==masterProcess){

     for(int i=0;i<numProcesses;i++){
         
         totalSum+=partialSumArr[i];
         cout<<"\nP"<<id<<" Received:"<<partialSumArr[i]<<"\n";	
     }

     cout<<"\nP"<<id<<"  TOTAL SUM: "<<totalSum;

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

