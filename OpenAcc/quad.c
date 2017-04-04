#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void getsubM(float* A, int n , int i, float* part);
//matrix multiplication
void matrixMult(float* m1 , float* m2, float* result, int c1 , int r1, int c2);

// print matrix for testing
void print(float *a, int row, int col){
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      printf("%4.2f ", a[col*i + j]);
    }
    printf("\n\n");
  }
}

//splits matrix A into three smaller matrices a1, a2, a3.
void getsubMs(float* A, float* A1, float* A2, float* A3 ,   int n ){
    int m = n/2;
    #pragma acc data copyin(A[0:n*n]) 
    {
        int i;
        int j;
        #pragma acc data copyout(A1[0:m*m])
        #pragma acc parallel loop collapse(2) 
        for( i = 0; i < m;i++ ){
            for(j = 0; j < m; j++){
                A1[m*i + j] = A[n*i + j];
            }
        }

        //Initializing A2(uv)
        int u = 0;
        int v;
        // if we were to use parallel here on our own we would need to have
        // a critical section of atomic updated
        // with the following parallel directives 
        #pragma acc data copyout(A2[0:m*m]) 
        #pragma acc kernels
        {  
            // #pragma acc parallel loop gang vector_length(m)
            for(i = 0; i < m; i++,u++ ){ 
                v = 0; 
                // #pragma acc parallel loop vector
                for(  j = m; j < n; j++,v++){
                    A2[m*u + v] = A[n*i + j];
                    // #pragma acc atomic update
                    v++;
                }
                // #pragma acc atomic update
                u++;
            }
        }
        
        int x = 0;
        #pragma acc data copyout(A3[0:m*m]) 
        #pragma acc kernels
        {   // #pragma acc parallel loop gang vector_length(m)
            for(i = m; i <n; i++ ){
                int y = 0;
                 // #pragma acc parallel loop vector
                for( j = m; j < n; j++){
                    A3[m*x + y] = A[n*i + j];
                    // #pragma acc atomic update
                    y++;
                }
                // #pragma acc atomic update
                x++;
            } 
        } 
    }
}

void matrixMult(float* m1 , float* m2, float* result, int c1 , int r1, int c2){
    int i,j;
    float sum;
    // Using the directive copyout we specifically state that the device variable at the start of the block will not copy the values of the host variable, but 
    // at the end of the block copy to the device to the host.    
    #pragma acc data copyin(m1[0:r1*c1]) copyin(m2[0:c1*c2])  copyout(result[0:r1*c2])
    #pragma acc parallel loop collapse(2) reduction(+:sum)
    for(i = 0; i < r1; ++i){
        // #pragma acc parallel loop reduction(+:sum) vector
        for( j = 0; j < c2; ++j){
            sum = 0;
            for(int k = 0; k < c1; ++k){
                sum  += m1[i*c1 + k]*m2[k*c2 + j];
            }
            result[i*c2 + j] =  sum;
        } 
    } 
} 

// vector multiplication. v1 and v2 are vectors to multiply, n is a size of
//a number, result is stored in scalar.
void vectorProduct(float* v1 , float* v2 , float* scalar , int n){
    int i;
    float sum = 0;
    #pragma acc data copyin(v1[0:n]) copyin(v2[0:n])
    #pragma acc parallel loop reduction(+:sum)
    for(i = 0; i < n; i++){
        sum +=  v1[i]*v2[i];
    }
    *scalar = sum;
} 

// finds quadratic form u'Au. A is a square matrix, n is number of rows in a, u is a vector.
float quad(float* a , int n , float* u){
    size_t sizeAi = (n/2)*(n/2)*sizeof(float);
    float* a1 = (float*)malloc(sizeAi);
    float* a2 = (float*)malloc(sizeAi);
    float* a3 = (float*)malloc(sizeAi);

    float* u1 = (float*)malloc((n/2)*sizeof(float));
    float* u2 = (float*)malloc((n/2)*sizeof(float));

    float* subResult = (float*)malloc((n/2)*sizeof(float));
    float num1 = 0;
    float num2 = 0;
    float num3 = 0;

    getsubMs(a,a1,a2,a3,n);

    for(int i = 0; i < n/2; ++i){
        u1[i] = u[i];
        u2[i] = u[(n/2) + i];
    }

    float sum = 0;

    //u1^t*a1
    matrixMult(u1,a1,subResult,n/2,1,n/2);
    vectorProduct(subResult,u1,&num1,n/2);
    matrixMult(u1,a2,subResult,n/2,1,n/2);
    vectorProduct(subResult,u2,&num2,n/2); 
    matrixMult(u2,a3,subResult,n/2,1,n/2);
    vectorProduct(subResult,u2,&num3,n/2);

    sum = num1 + 2*num2 + num3;

    free(u1);
    free(u2);
    free(a1);
    free(a2);
    free(a3);
    free(r1);
    return sum;
} 

//takes in one argument: number of rows for a symmetric matrix.
int main(int argc, char* argv[]){
    int n = atoi(argv[1]);
    float* a = (float*)malloc(n*n*sizeof(float));
    float* u = (float*)malloc(n*sizeof(float));
    for(int i = 0; i < n; i++){
        u[i] = i + 1;
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < i; j++){
            a[i*n+j] = i;
            a[j*n+i] = i;
        }
    }

    for(int k = 0; k < n; k++){
        a[k*n+k] = k + 1;
    }

    float result = quad(a,n,u);
    printf("result:%f \n",result);
} 