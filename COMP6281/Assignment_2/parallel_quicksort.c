#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// function to swap elements
void swap(int *a, int *b) {
  int t = *a;
  *a = *b;
  *b = t;
}
// Comparison function for qsort
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int selectPivot(int arr[], int n) {
    // Sort the array in ascending order
    qsort(arr, n, sizeof(int), compare);
    return arr[n/2];
}

// function to find the partition position
int partitionArr(int array[], int n, int pivot, int results[]) {
  int j;
  int l = 0;
  int r = n-1;
  // traverse each element of the array
  // compare them with the pivot
  for (int j = 0; j < n; j++) {
    if (array[j] < pivot) {
      results[l] = array[j];
      l++;
    }else{
      results[r] = array[j];
      r--;
    }
  }
  return l;
}
 
int partitionQuicksort(int arr[], int low, int high) {  
    int pivot = arr[high];  
    int i = (low - 1);  
  
    for (int j = low; j <= high - 1; j++) {  
        if (arr[j] < pivot) {  
            i++;  
            swap(&arr[i], &arr[j]);  
        }  
    }  
    swap(&arr[i + 1], &arr[high]);  
    return (i + 1);  
}  

void quickSort(int arr[], int low, int high) {  
    if (low < high) {  
        int pi = partitionQuicksort(arr, low, high);  
        quickSort(arr, low, pi - 1);  
        quickSort(arr, pi + 1, high);  
    }  
}  

// Function to split an array into n chunks
void splitArray(int arr[], int size, int n, int* result[], int result_size[]) {
    int chunk_size = size / n;
    int remainder = size % n;
    int index = 0;

    for (int i = 0; i < n; i++) {
        int current_chunk_size = chunk_size;

        // Distribute the remainder if it exists
        if (remainder > 0) {
            current_chunk_size++;
            remainder--;
        }

        // Allocate memory for the current chunk
        result[i] = (int*)malloc(current_chunk_size * sizeof(int));
        result_size[i] = current_chunk_size;

        // Copy elements into the current chunk
        for (int j = 0; j < current_chunk_size; j++) {
            result[i][j] = arr[index++];
            //printf( "%d,", result[i][j]);
        }
    //printf( "\n");
    }
}

int main(int argc, char* argv[])
{   
    int numParents, processId, i, pivot, partition, localRank, j, color, split;
    float endTime, startTime;
    
    // set max, min numnber in random number generator + size of array to sort
    int size = (int)pow(2, 20);

    MPI_Comm childComm;
    
    // Obtain number of tasks and task ID 
    MPI_Init(&argc,&argv);

    startTime = MPI_Wtime ( );

    MPI_Comm_size(MPI_COMM_WORLD, &numParents);

    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    // this is the initial shard. this will change throughout program
    // execution
    int shard = size/numParents;
    int cubeDim = log2(numParents);
    int myint;

    // local array needs to be dynamic - not guaranteed to be this value
    int *localA = NULL;
    int *arr = NULL;
    localA = (int *)malloc(shard * sizeof(int));

    // initial startup sequence, broadcast the 
    if (processId==0){
      arr = (int *)malloc(size * sizeof(int));
      // the number of items we have in our list
      srand(0);
      for (i = 0; i < size; i++) {
        arr[i] = rand();
      }

      // Create an array of pointers to hold the chunks
      int* result[numParents];
      int result_size[numParents];

      // Split the array into chunks
      splitArray(arr, size, numParents, result, result_size);

        for (int i = 1; i < numParents; i++) {
          // break the array, sending to child process 
          MPI_Send(result[i], size/numParents, MPI_INT, i, 0, MPI_COMM_WORLD);
        } 
      
      // assign the localA for the master process, treat it as the other
      // processes for the next step
      for (int i = 0; i < shard; i++) {
        localA[i] = result[0][i];
      }
      free(arr);
    }else{
      MPI_Recv(localA, shard, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // put a barrier here so that we chunk the array before next step
    //MPI_Barrier(MPI_COMM_WORLD);

    /*printf("I am rank %d and here is my array:", processId);
    for (int i = 0; i < shard; i++) {
      printf("%d,", localA[i]);
    }
    printf("\n");
    */
    int lenSubArr = shard;

    for (int i = 0; i < cubeDim; i++) {

      // pivot selection process, happens in first process of each partition
      partition = pow(2,(cubeDim-i));
      //printf("The partition is %d\n", partition);

      color = processId/partition;
      int localRank;

      MPI_Comm partitionComm;
      MPI_Comm_split(MPI_COMM_WORLD, color, 0, &partitionComm);

      MPI_Comm_rank(partitionComm, &localRank);

      if (localRank == 0){
        pivot = selectPivot(localA, lenSubArr);
      }

      MPI_Bcast(&pivot, 1, MPI_INT, 0, partitionComm);

      // partition every array arround the local pivot
      int results[lenSubArr];
      split = partitionArr(localA, lenSubArr, pivot, results);

      int lenLower = split;
      int lower[lenLower];
      for (int j = 0; j<lenLower; j++){
        lower[j] = results[j];
        //printf("lower item %d for process %d: %d, pivot: %d\n", j, processId, results[j], pivot);
      }

      int lenUpper = lenSubArr-split;
      int upper[lenUpper];
      for (int j = 0; j<(lenUpper); j++){
        upper[j] = results[lenLower+j];
        //printf("upper item %d for process %d: %d, pivot: %d\n", j, processId, results[split+j], pivot);
      }
      //printf("The local pivot in process %d is %d\n", processId, pivot);

      // now that we have our array, we want to split the color (processor group) in half, sending 
      // the small items up and the large items down (i.e. if you have 4 processors in the group ,
      // we are going to send 2->0, 0->2, 3->1 and 1->3).
      // if there are 8 total processes, on first step all color is 0
      // localRank = (0,1,2,3,4,5,6,7) halfRank = (0,0,0,0,1,1,1,1) halfOrder = (0,1,2,3,0,1,2,3)
      // halfRank = (0,0,0,0,1,1,1,1), halfOrder = (0,1,2,3,0,1,2,3), partition = (0,0,0,0,0,0,0,0)
      int half = partition/2;
      int halfRank = localRank/half;
      int halfOrder = localRank%half;
      int recipient = (1-halfRank)*half+halfOrder;

      if (i==-1){
        printf("i am process %d, i have halfrank %d and halforder %d. I am sending to local process %d\n", processId, halfRank, halfOrder, recipient);
      }

      int nRecv;

      if (halfRank==1){
        MPI_Send(&lenLower,1, MPI_INT, recipient, 0, partitionComm);
      }else{
        MPI_Recv(&nRecv, 1, MPI_INT, recipient, 0, partitionComm, MPI_STATUS_IGNORE);
      }

      if (halfRank==0){
        MPI_Send(&lenUpper,1, MPI_INT, recipient, 0, partitionComm);
      }else{
        MPI_Recv(&nRecv, 1, MPI_INT, recipient, 0, partitionComm, MPI_STATUS_IGNORE);
      }
      //printf("I am process %d, I am going to recieve %d items\n", processId, nRecv);
      MPI_Barrier(MPI_COMM_WORLD);

      int recvArr[nRecv];

      if (halfRank==1){
        MPI_Send(lower, lenLower, MPI_INT, recipient, 0, partitionComm);
        //printf("Finished sending from halfrank 1\n");
      }else{
         MPI_Recv(&recvArr, nRecv, MPI_INT, recipient, 0, partitionComm, MPI_STATUS_IGNORE);
      }

      if (halfRank==0){
        MPI_Send(upper, lenUpper, MPI_INT, recipient, 0, partitionComm);
        //printf("Finished sending from halfrank 1\n");
      }else{
         MPI_Recv(&recvArr, nRecv, MPI_INT, recipient, 0, partitionComm, MPI_STATUS_IGNORE);
      }

      if (halfRank==1){
        lenSubArr = lenUpper+nRecv;
      }else{
        lenSubArr = lenLower+nRecv;
      }
      localA = (int *)realloc(localA, lenSubArr * sizeof(int));

      if (halfRank==1){
        for (j=0;j<lenUpper;j++){
          localA[j] = upper[j];
        }
        for (j=0;j<nRecv;j++){
          localA[j+lenUpper] = recvArr[j];
        }
      }else{
        for (j=0;j<lenLower;j++){
          localA[j] = lower[j];
        }
        for (j=0;j<nRecv;j++){
          localA[j+lenLower] = recvArr[j];
        }
      }
    MPI_Barrier(partitionComm);
    MPI_Comm_free(&partitionComm);
    }

    // one of the final, very important steps. quicksort the local array
    quickSort(localA, 0, lenSubArr-1);

    //printf("I am process %d, here is my array after recv/send: ", processId);
    //for (j=0;j<lenSubArr;j++){
    //  printf("%d,", localA[j]);
    //}
    //printf("\n");


    MPI_Status newStatus;
    // send, for all the processes, save the main process
    if (processId!=0){
       MPI_Send(&lenSubArr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    int *disPlace = NULL;
    int *recvSize = NULL;
    int *finalArr = NULL;

    if (processId==0){
      disPlace = malloc( numParents * sizeof(int) );
      recvSize = malloc( numParents * sizeof(int) );
      finalArr = malloc(size * sizeof(int));

      disPlace[0]=0;
      recvSize[0]=lenSubArr;
      for (i=1;i< numParents; i++){
        int nRecv;
        MPI_Recv(&nRecv, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        disPlace[i]=disPlace[i-1]+recvSize[i-1];
        recvSize[i]=nRecv;
      }
    }

    // shows the displacements and recviecing counts
    /*
    if (processId==0){
      for (i=0;i<n;i++){
        printf("%d,",disPlace[i]);
      }
      printf("\n");
      for (i=0;i<n;i++){
        printf("%d,",recvSize[i]);
      }
      printf("\n");
    }*/

    MPI_Gatherv(localA, lenSubArr, MPI_INT, finalArr, recvSize, disPlace, MPI_INT, 0, MPI_COMM_WORLD);

    //MPI_Barrier(MPI_COMM_WORLD);
    if (processId==0){
      //printf("The final array is as follows:\n");
      //for (i=0; i<size;i++){
      //  printf("%d,",finalArr[i]);
      //}
      //printf("\n");
    }

    free(localA);
    free(disPlace);
    free(recvSize);
    free(finalArr);
    
    endTime = MPI_Wtime ( ) - startTime;

    if ( processId == 0 )
    {
     printf("Elapsed time =  %f seconds.\n",endTime);	
    }

    MPI_Finalize();
    return 0;

}