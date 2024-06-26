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
    int pi;
    if (low < high) {  
        //printf("iterate");
        pi = partitionQuicksort(arr, low, high);  
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
    int size = (int)pow(2, 20);
    int *arr = NULL;

    MPI_Comm childComm;
    
    // Obtain number of tasks and task ID 
    MPI_Init(&argc,&argv);

    startTime = MPI_Wtime ( );

    MPI_Comm_size(MPI_COMM_WORLD, &numParents);

    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    int shard = (int)size/numParents;
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

    int *localSamples = NULL;
    localSamples = (int *)malloc(numParents * sizeof(int));
    for (i=0;i<numParents;i++){
        int idx = (int)i*size/pow(numParents,2);
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
    free(localSamples);

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

    //MPI_Barrier(MPI_COMM_WORLD);

    // define the local lengths that will be used for gatherv
    int recvBuffer[numParents*numParents];
    MPI_Allgather(arrLens, numParents, MPI_INT, recvBuffer, numParents, MPI_INT, MPI_COMM_WORLD);

    //if (processId==0){
    //    printf("All array lengths, broadcast to all threads:");
    //    for (i=0;i<numParents*numParents;i++){
    //        printf("%d,",recvBuffer[i]);
    //    }
    //    printf("\n");
    //}
    //printf("\n");
    int *localGatherV = NULL;
    int *localCounts = NULL;
    int *localDisplace = NULL;
    int *sendArray = NULL;
    int startIdx;

    //printf("process %d, arrLens = ", processId);
    //for (j=0;j<numParents;j++){
    //    printf("len = %d, bound=%d, ", arrLens[j],arrBounds[j]);
    //}
    //printf("\n");

    for (i=0;i<numParents;i++){
        if (processId==i){    
            int count=0; 
            localCounts = (int *)malloc(numParents * sizeof(int)); 
            localDisplace = (int *)malloc(numParents * sizeof(int));   
            localDisplace[0]=0;
            localCounts[0] = recvBuffer[i];
            count+=recvBuffer[i];
            //printf("Local count: %d, local displace %d, process %d\n", localCounts[0], localDisplace[0], processId);
            for (j=1;j<numParents;j++){
                localCounts[j] = recvBuffer[j*numParents+i];
                localDisplace[j] = localDisplace[j-1] + localCounts[j-1];
                count+=recvBuffer[j*numParents+i];
                //printf("Local count: %d, local displace %d, process %d\n", localCounts[j], localDisplace[j], processId);
            }

            localGatherV = (int*)malloc(count * sizeof(int));
        }
        //for (j=0;j<numParents;j++){
        //    printf("%d", arrLens[j]);
        //}

        sendArray = (int*)malloc(arrLens[i] * sizeof(int));

        if (i==0){
            startIdx=0;
        }else{
            startIdx=arrBounds[i-1];
        }
        //printf("startIdx=%d", startIdx);
        for (j=0;j<arrLens[i];j++){
            sendArray[j]=localA[startIdx+j];
            //printf("%d,", sendArray[j]);
        }
        //if (processId==0){
        //    printf("I am process 0, sending to process %d, the following array, starting at bounds %d: ", i, startIdx);
        //    for (j=0;j<arrLens[i];j++){
                //sendArray[j] = localA[arrBounds[i-1]+j];
        //        printf("%d,", sendArray[j]);
        //    }
        //}
        //printf("\n");

        if (processId==i){
            MPI_Gatherv(sendArray,arrLens[i],MPI_INT,localGatherV,localCounts,localDisplace,MPI_INT,i,MPI_COMM_WORLD);
        }else{
            MPI_Gatherv(sendArray,arrLens[i],MPI_INT,NULL,NULL,NULL,MPI_INT,i,MPI_COMM_WORLD);
        }
    }
    free(sendArray);
    free(localA);

    if (processId==0){
        //free(globalSamples);
        free(recvSize);
        free(disPlace);
    }

    int totalCount = 0;
    for (j=0;j<numParents;j++){
        totalCount+=localCounts[j];
    }
    //printf("Total count going in to final quicksort is: %d (process %d)\n", totalCount, processId);
    
    timeStart = MPI_Wtime();
    quickSort(localGatherV, 0, totalCount-1);
    timeEnd = MPI_Wtime() - timeStart;

    //printf("Finished quicksort of local array, time elapsed = %f, len %d\n", timeEnd,totalCount);

    int *sortedArr = NULL;
    int *finalArrLens = NULL;
    int *finalDisplace = NULL;
    if (processId==0){
        finalArrLens = (int*)malloc(numParents * sizeof(int));
        finalDisplace = (int*)malloc(numParents * sizeof(int));
    }
    
    MPI_Gather(&totalCount,1,MPI_INT,finalArrLens,1,MPI_INT,0,MPI_COMM_WORLD);
    
    if (processId==0){
        finalDisplace[0]=0;
        for (i=1;i<numParents;i++){
            finalDisplace[i]=finalArrLens[i-1]+finalDisplace[i-1];
        }
        sortedArr = (int*)malloc(size * sizeof(int));
    }

    MPI_Gatherv(localGatherV,totalCount,MPI_INT,sortedArr,finalArrLens,finalDisplace,MPI_INT,0,MPI_COMM_WORLD);
    
    
    /*if (processId==0){
        printf("FINAL ARRAY:\n");
        for (i=0;i<size;i++){
            printf("%d,",sortedArr[i]);
        }
        printf("\n");
    }*/

    //MPI_Barrier(MPI_COMM_WORLD);

    endTime = MPI_Wtime ( ) - startTime;

    if ( processId == 0 )
    {
     printf("Elapsed time =  %f seconds.\n",endTime);	
    }

    if (processId==0){
        free(sortedArr);
        free(finalArrLens);
        free(finalDisplace);
    }

    MPI_Finalize();
    return 0;
}