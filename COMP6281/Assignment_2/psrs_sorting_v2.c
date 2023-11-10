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


// Function to partition the array into p segments based on given boundaries
// imagine we have [2, 5, 8, 15, 21, 22, 34, 35, 37]
// partitions are [1, 17, 35, 40]
// want 
void partitionArray(int arr[], int n, int p, int boundaries[], int arrBounds[]) {
    if (p <= 1 || n < p) {
        printf("Invalid input for partitioning.\n");
        return;
    }
    int i,j;
    j=0;
    for (i=0; i<n;i++){
        //printf("j,%d = i %d\n", j,i);
        while (1) {
            if (arr[i]<boundaries[j]){
                break;
            }else{
                arrBounds[j]=i;
                if (j==(p-1)){
                    break;
                }else{
                    j++;
                }
            }
        }
    }
    while (j<p){
        arrBounds[j]=i;
        j++;
    }
}

int main(int argc, char* argv[])
{      
    srand(0);
    int numParents, processId, i, pivot, partition, localRank, j, color, split;
    float endTime, startTime;
    float timeStart,timeEnd;

    int *localA = NULL;
    int *globalSamples = NULL;
    int *recvSize = NULL;
    int *disPlace = NULL;
    int *resultSize = NULL;

    // set max, min numnber in random number generator + size of array to sort
    int size = (int)pow(2,18);
    int *arr = NULL;

    MPI_Comm childComm;
    
    // Obtain number of tasks and task ID 
    MPI_Init(&argc,&argv);

    startTime = MPI_Wtime ( );

    MPI_Comm_size(MPI_COMM_WORLD, &numParents);

    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    int shard = size/numParents;
    int pivots[numParents-1];

    // local array needs to be dynamic - not guaranteed to be this value
    localA = (int *)malloc(shard * sizeof(int));
    //printf("Allocating localA in process %d, size %d\n", processId, shard);

    if (processId==0){
      arr = (int *)malloc(size * sizeof(int));
      //printf("Allocating total array in main, size %d\n", size);

      timeStart = MPI_Wtime();
      // the number of items we have in our list
      for (i = 0; i < size; i++) {
        arr[i] = rand();
      }
      timeEnd = MPI_Wtime() - timeStart;
      //printf("Finished generating random array, time elapsed = %f\n", timeEnd);

      // Create an array of pointers to hold the chunks
      int* result[numParents];
      int resultSize[numParents];

      // Split the array into chunks
      timeStart = MPI_Wtime();
      splitArray(arr, size, numParents, result, resultSize);
      timeEnd = MPI_Wtime() - timeStart;
      //printf("Finished splitting random array, time elapsed = %f\n", timeEnd);
      //for (i=0;i<numParents;i++){
      //  printf("size of split array: %i\n", resultSize[i]);
      //}

        for (int i = 1; i < numParents; i++) {
          // break the array, sending to child process 
          timeStart = MPI_Wtime();
          MPI_Send(result[i], shard, MPI_INT, i, 0, MPI_COMM_WORLD);
          timeEnd = MPI_Wtime() - timeStart;
          //printf("Finished sending arrays to children, time elapsed = %f\n", timeEnd);
        } 
      
      // assign the localA for the master process, treat it as the other
      // processes for the next step
      for (int i = 0; i < shard; i++) {
        localA[i] = arr[i];
      }
      free(arr);
    }else{
      MPI_Recv(localA, shard, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    timeStart = MPI_Wtime();
    quickSort(localA, 0, shard-1);
    timeEnd = MPI_Wtime() - timeStart;
    //printf("Finished quicksort of local array, time elapsed = %f\n", timeEnd);

    //MPI_Finalize();
    //return 0;
    int localSamples[numParents];
    for (i=0;i<numParents;i++){
        int idx = i*size/pow(numParents,2);
        //printf("%d,",idx);
        localSamples[i] = localA[idx];
    }

    //printf("\n");

    // total random samples to gather
    int nSamples = pow(numParents,2);
    
    if (processId==0){
        globalSamples = malloc( nSamples * sizeof(int) );
        recvSize = malloc( numParents * sizeof(int) );
        disPlace = malloc( numParents * sizeof(int) );
        
        recvSize[0] = numParents;
        disPlace[0] = 0;
        for (i=1;i<numParents;i++){
            recvSize[i] = numParents;
            disPlace[i] = recvSize[i-1]+disPlace[i-1];
        }
    }

    MPI_Gatherv(localSamples, numParents, MPI_INT, globalSamples, recvSize, disPlace, MPI_INT, 0, MPI_COMM_WORLD);

    if (processId==0){
        quickSort(globalSamples,0,nSamples-1);
        
        //for (i=0;i<nSamples;i++){
        //    printf("%d,",globalSamples[i]);
        //}

        for (i=0;i<numParents-1;i++){
            int index = numParents*(i+1)+numParents/2-1;
            pivots[i] = globalSamples[index];
            //printf("%d,",pivots[i]);
        }
    }

    MPI_Bcast(pivots,numParents-1,MPI_INT,0,MPI_COMM_WORLD);

    //printf("Process %d: ", processId);
    //for (i=0;i<numParents-1;i++){
    //    printf("%d,",pivots[i]);
    //}
    //printf("\n");
    
    int arrBounds[numParents-1];
    int arrLens[numParents];

    //printf("breaking a shard of lenth %d into %d parts\n", shard, numParents);
    partitionArray(localA, shard, numParents, pivots, arrBounds);

    arrLens[0] = arrBounds[0];
    arrLens[numParents-1] = shard-arrBounds[numParents-2];
    for (i=1;i<numParents-1;i++){
        arrLens[i] = arrBounds[i]-arrBounds[i-1];
        
    }
    //if (processId==0){
    //    printf("Process 0 bounds: ");
        // 11,16,18,19,
        // 11, 27, 45 
    //    for (i=0;i<numParents-1;i++){
    //        printf("%d,", arrBounds[i]);
    //    }
    //    printf("\n");
    //}

    // define the local lengths that will be used for gatherv. the recv buffer shows the length 
    // of each array in the sub array next to each partition
    int recvBuffer[numParents*numParents];
    MPI_Allgather(arrLens, numParents, MPI_INT, recvBuffer, numParents, MPI_INT, MPI_COMM_WORLD);

    if (processId==-1){
        printf("All array lengths, broadcast to all threads:");
        for (i=0;i<numParents*numParents;i++){
            printf("%d,",recvBuffer[i]);
        }
        printf("\n");
    }
    //printf("\n");
    // send the local array to all other arrays

    int globalRecvArr[size];
    int allShards[numParents];
    int allShardDisplace[numParents];

    allShards[0] = shard;
    allShardDisplace[0] = 0;
    for (i=1; i<numParents; i++){
        allShards[i] = shard;
        allShardDisplace[i] = allShardDisplace[i-1] + allShards[i-1];
    }
    MPI_Allgatherv(localA, shard, MPI_INT, globalRecvArr, allShards, allShardDisplace, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    int localShards[numParents];
    int totalLocal = 0;
    int localDisplace[numParents*numParents];
    localDisplace[0] = 0;
    for (i=0; i<numParents; i++){
        for (j=0; j<numParents; j++){
            if (i+j==0){
                continue;
            }
            localDisplace[(i*numParents)+j] = localDisplace[(i*numParents)+j-1]+recvBuffer[(i*numParents)+j-1];
        }
        localShards[i] = recvBuffer[i*numParents + processId];
        totalLocal += localShards[i];
    }
    if (processId==-1){
        for (i=0;i<numParents*numParents;i++){
            printf("%d,", localDisplace[i]);
        }
    }

    int finalLocalArr[totalLocal];
    int total=0;
    for (i=0; i<numParents; i++){
        int start = localDisplace[i*numParents+processId];
        int stop = start + localShards[i];
        for (j=start; j<stop; j++){
            finalLocalArr[total] = globalRecvArr[j];
            total++;
        }
    }
    // this is the bottleneck. we gotta speed this up!!
    quickSort(finalLocalArr, 0, total-1);

    
    //MPI_Barrier(MPI_COMM_WORLD);
    int *sortedArr = NULL;
    int *finalArrLens = NULL;
    int *finalDisplace = NULL;
    if (processId==0){
        finalArrLens = (int*)malloc(numParents * sizeof(int));
        finalDisplace = (int*)malloc(numParents * sizeof(int));
    }
    
    MPI_Gather(&totalLocal,1,MPI_INT,finalArrLens,1,MPI_INT,0,MPI_COMM_WORLD);
    
    if (processId==0){
        finalDisplace[0]=0;
        for (i=1;i<numParents;i++){
            finalDisplace[i]=finalArrLens[i-1]+finalDisplace[i-1];
        }
        sortedArr = (int*)malloc(size * sizeof(int));
    }

    MPI_Gatherv(finalLocalArr,totalLocal,MPI_INT,sortedArr,finalArrLens,finalDisplace,MPI_INT,0,MPI_COMM_WORLD);
    //if (processId==0){
    //    for (i=0;i<size;i++){
    //        printf("%d,", sortedArr[i]);
    //    }
    //}
    endTime = MPI_Wtime ( ) - startTime;

    if ( processId == 0 )
    {
     printf("Elapsed time =  %f seconds.\n",endTime);	
    }
    
    MPI_Finalize();
    return 0;
}