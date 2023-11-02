#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

int main(int argc, char* argv[])
{   
    int i;
    clock_t startTime, endTime;

    // set max, min numnber in random number generator + size of array to sort
    int maximum_number = (int)pow(2,16);;
    int minimum_number = 0;
    int size = (int)pow(2,16);;
    int arr[size];

    startTime = clock();

    for (i = 0; i < size; i++) {
        arr[i] = rand() % (maximum_number + 1 - minimum_number) + minimum_number;
    }

    quickSort(arr, 0, size-1);

    //printf("FINAL ARRAY:\n");
    //for (i=0;i<size;i++){
    //    printf("%d,",arr[i]);
    //}
    //printf("\n");

    endTime = clock() - startTime; 
    double doubleTime = ((double)endTime)/CLOCKS_PER_SEC; // in seconds 
    printf("Elapsed time =  %f seconds.\n",doubleTime);	

    return 0;

}