#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

void nmf(float *a, int r, int c , int k , int niters, float *w, float *h);
void matrixMult(float* m1 , int r1, int c1, float* m2, int r2 , int c2, float* result );
void elementMult(float* m1,float* m2, int r, int c);
void elementDiv(float* m1 , float* m2 , int r, int c);
void initMatrixH(float* mat, int r, int c, int k,float* h);
void initMatrixW(float* mat, int r , int c, int k,float* w);
void transpose( float* mat,float* t, int r , int c);
float max( float* mat,int r,int c);
void init(float* A , int r , int c);
void print(float* A, int r, int c);

//takes in number of rows, number of columns, approximation rank, and number of iterations.
int main(int argc,char* argv[]){
	int r = atoi(argv[1]);
	int c = atoi(argv[2]);
	int k = atoi(argv[3]);
	int niters = atoi(argv[4]);

	float* a = (float*)malloc(r*c*sizeof(float));
	// initial matrix a
	init(a,r,c);
	float *h = NULL;
	float *w = NULL;
	printf("current A\n");
	//print the initial matrix
	print(a,r,c);

	nmf(a,r,c,k,niters,w,h);
	//print the approximation
	printf("new A\n");
	print(a,r,c);
} // end main()

void nmf(float *a , int r , int c , int k, int niters, float* w, float *h){
	size_t sizeA = r*c*sizeof(float);
	size_t sizeW = r*k*sizeof(float);
	size_t sizeH = k*c*sizeof(float);
	size_t sizeRxK = r*k*sizeof(float);
	
	w = (float*)malloc(sizeW);// dim: rxk
	h = (float*)malloc(sizeH);// dim: kxc
	//memory allocation
	float* wT = (float*)malloc(sizeW);
	float* hT = (float*)malloc(sizeH);
	float* oldw = (float*)malloc(sizeW);

	float* updateWNum = (float*)malloc(sizeRxK);
	float* updateWDem1 = (float*)malloc(sizeA);
	float* updateWDem2 = (float*)malloc(sizeRxK);
	float* updateHNum = (float*)malloc(sizeH);
	float* updateHDem1 = (float*)malloc(k*k*sizeof(float));
	float* updateHDem2 = (float*)malloc(sizeH);
	//create matrices W and H
	initMatrixW(a,r,c,k,w);
	initMatrixH(a,r,c,k,h);

	for( int i = 0; i < niters; ++i){
		// since w is updated first
		memcpy(oldw,w,sizeW);
		// updating W
		transpose(h,hT,k,c); // h^t
		// a*h^t
		matrixMult(a,r,c,hT,c,k,updateWNum); // returns rxk
		// w*h
		matrixMult(w,r,k,h,k,c,updateWDem1); // returns rxc
		// (w*h)*h^t
		matrixMult(updateWDem1,r,c,hT,c,k,updateWDem2); // returns rxk
		// (a*h^t)/(w*h)*h^t
		elementDiv(updateWNum,updateWDem2,r,k); // return in updateWNum
		// computes new W
		elementMult(w,updateWNum,r,k);

		// updating H
		transpose(oldw,wT,r,k);  // w^t
		// w^t*a
		matrixMult(wT,k,r,a,r,c,updateHNum); // returns kxc
		// w^t*w
		matrixMult(wT,k,r,w,r,k,updateHDem1); // returns kxk
		// // (w^t*w)*h
		matrixMult(updateHDem1,k,k,h,k,c,updateHDem2); // returns kxc
		// (w^t*a)/(w^t*w*h)
		elementDiv(updateHNum,updateHDem2,k,c); // returns to updateHNum
		// computes new H
		elementMult(h,updateHNum,k,c);
	} 

	// computes new A
	matrixMult(w,r,k,h,k,c,a);

	free(wT);
	free(hT);
	free(oldw);

	free(updateWNum );
	free( updateWDem1 );
	free( updateWDem2 );
	free(updateHNum ); 
	free( updateHDem1);
	free(updateHDem2); 
} // end nmf()

// print the matrix for testing
void print(float* A, int r,int c){
	for(int i = 0; i < r; i++){
		for(int j = 0; j < c; j++){
			printf("%f ",A[c*i + j]);
		}
		printf("\n");
	}
	printf("\n");
}

// matrix multiplication
void matrixMult(float* m1 , int r1, int c1, float* m2, int r2 , int c2, float* result ){
	int i,j;
	float sum;
	// Using the directive copyout we specifically state that the device variable
	// at the start of the block will not copy the values of the host
	// variable, but at the end of the block copy the device to the host.    
	#pragma acc data copyin(m1[0:r1*c1]) copyin(m2[0:r2*c2]) copyout(result[0:r1*c2])
	#pragma acc parallel loop collapse(2) reduction(+:sum)
	for(i = 0; i < r1; ++i){
		for( j = 0; j < c2; ++j){
			sum = 0;
			for(int k = 0; k < c1; ++k){
				sum  += m1[i*c1 + k]*m2[k*c2 + j];
			}
			result[i*c2 + j] =  sum;
		} 
	} 
} 

void elementMult(float* m1 , float* m2, int r , int c){
	int i;
	#pragma acc data copyin(m2[0:r*c]) copy(m1[0:r*c])
	#pragma acc parallel loop 
	for(i = 0; i < r*c; ++i){
		m1[i] = m1[i]*m2[i];
	}
} 

// matrix element by element division. m1 is a dividend, m2 is a divisor, r 
// is number of rows, c is number of columns. The result is stored in m1.
void elementDiv(float* m1 , float* m2, int r ,int c){
	int i;
	#pragma acc data copyin(m2[0:r*c]) copy(m1[0:r*c])
	#pragma acc parallel loop 
	for(i = 0; i < r*c; ++i){
		m1[i] = m1[i]/m2[i];
	}
}

// matrix transpose. Takes in the matrix to transpose, mat, the matrix to 
// store result, t, number of rows and number of columns, r and c 
// respectively.
void transpose( float* mat , float* t, int r , int c){
	int i,j;
	#pragma acc data copyin(mat[0:r*c]) pcopyout(t[0:r*c])
	#pragma acc parallel loop collapse(2) 
	for(i = 0; i < r; i++){
		for( j = 0; j < c; j++){
			t[r*j + i] = mat[c*i + j];
		}
	}

}

// create initial matrix A
void init (float* A, int r, int c){
	for(int i = 0; i < r*c; i++){
		A[i] = i + 1;
	}
}

// Find the biggest element in a matrix
float max(float* mat,int r,int c){
    float mx = 0;
    int i;
    for(i = 0; i < r*c;i++){
        if (mx < mat[i]){
            mx = mat[i];
        }
    }
    return mx;
}

//create random matrix W. 
void initMatrixW(float* mat, int r , int c, int k, float* w){
    float mx = sqrt(max(mat,r,c));
    int n = k*r;
    time_t t;
    srand((unsigned)time(&t));
    int i; 
    for( i = 0; i < n; i++){
        w[i] = (rand() % (int)mx + 1);
    }
}

// create random matrix H
void initMatrixH(float* mat, int r , int c, int k, float * h){
    float mx = max(mat,r,c);
    int n = k*c;
    time_t t;
    srand((unsigned)time(&t));
    int i;
 
    for(i = 0; i < n; i++){
        h[i] = (float)( rand() % (int)(mx + 1) );
    }
} 