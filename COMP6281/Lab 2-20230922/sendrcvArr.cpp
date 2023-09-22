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

  int *sendArr; //Pointers only as array allocation will be dynamic
  int *recvArr;
  
  int count=3;  //Number of elements each process will receive

  int source=0;


  const int tag=10;

//  Initialize MPI.

  MPI_Init ( &argc, &argv );  

  startTime = MPI_Wtime ( );

//  Get the number of processes.

  MPI_Comm_size ( MPI_COMM_WORLD, &numProcesses );

//  Get the individual process ID.

  MPI_Comm_rank ( MPI_COMM_WORLD, &id );


// ID 0 is the master here. It dynamically populates an array of size numProcesses*count(3 in this case) and sends it to all processes. It sends it to itself as well.

  if ( id == 0 ) {

    sendArr= new int[numProcesses*count];

    //Populate Array
    for(int i=0;i<numProcesses*count;i++){
	sendArr[i]=i;
    }

   //Send 3 elements to each process

    for(int i=0;i<numProcesses;i++){

	MPI_Send(&sendArr[i*count],count,MPI_INT,i,i,MPI_COMM_WORLD); //Notice the tag changes every time as well. In Recv, we use MPI_ANY_TAG.
	
        //BE CAREFUL ABOUT POINTERS...Should put the & sign.

    }

  }


  //else{
	
    recvArr=new int[count];

    MPI_Recv(recvArr,count,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   //USE OF Wildcarding for source. 
    
    for(int i=0;i<count;i++){

    cout << "\nP" << id << ":  Value received: " << recvArr[i];

    }
    cout<<"\n\n";
  //}




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

