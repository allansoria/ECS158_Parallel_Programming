#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// print the matrix for error checking
void print(float *a, int row, int col){
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      printf("%4.2f ", a[col*i + j]);
    }
    printf("\n");
  }
}

// generate a random value between 0 and 1
float r2(){
  return (float)rand() / (float)RAND_MAX;
}
 

void filter(float *a,int rows, float b){
  int flag = 0;
  int cols = rows;
  float cn = 0;
  int i;
  int j;
  int n = cols*rows;
  // Within the matrix at every row we record a consecutive count of successfully being within the threshold 
  #pragma acc data copy(a[0:n]) 
  {
    #pragma acc parallel loop  reduction(+:cn)
    for( i = 0; i < rows; i++){
      cn = 0;
      #pragma acc parallel loop 
      for(j = 0; j < cols; j++){
        if(a[cols*i+j] > b){
          cn++;  
        }
        else{
          cn = a[cols*i + j];
       }
      
         a[cols*i + j] = cn;
      }
    }
  }

}

// checks areas of size k for a minimum brightness and returns the number
// of spots that fit the criteria
int findBright(float *a, int rows, int k){
  int cols = rows;
  int  brightspot = 0;
  int flag;
  int coljump;
  int n = cols*rows;
  int i,j;

  #pragma acc data pcopyin(a[0:n]) 
  { 
    #pragma acc parallel loop collapse(2) reduction(+:brightspot)
    for( i = 0; i < (rows-(k-1)); i++){
      for(j = k-1; j < cols; j++){
         if(a[cols*i+j] >= k){
           flag = 1;

           for(coljump = 0; coljump < k-1; coljump++){
             if(a[(cols*(i+(coljump+1))+j)] < k){
               flag = 0;
             }
           }
           if(flag == 1){
              brightspot++;
           }
        }
      }
    }
}
  return brightspot;
}


int brights(float *pix,int n, int k, float thresh){
  filter(pix,n,thresh);
  return findBright(pix,n,k);
}

int main(int argc, char* argv[]){
  int rows = atoi(argv[1]);
  int k = atoi(argv[2]);
  float t = atof(argv[3]);
  int cols = rows;
 
  float* a = (float*)malloc(cols*rows*sizeof(float));

  // create a matrix of size argv[1] by argv[1] with random values between 0 and 1 
  for(int i = 0; i < rows*cols; i++){
    a[i] = (float)r2();
  }
  
  // start timer (tracks CPU time, NOT elapsed time)
  clock_t start = clock(), diff;

  printf("brightspots: %d\n",brights(a,rows,k,t));

  // end timer
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("\nTime taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
} 