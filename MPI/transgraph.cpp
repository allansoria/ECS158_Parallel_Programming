# include <mpi.h>
# include <stdlib.h>
# include <stdio.h>

int myrows[2];
int total[1];

int NNodes,Me;
MPI_Status status;
int temp[1];

#define SENDROWS 0
#define SENDADJM 1
#define SENDRESULT 2
#define SENDRESULT1 3
#define SENDRESULT2 4

void print(int* adjm , int r,int c, int n){
  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){
      printf("%d ",adjm[i*n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void findmyrange(int n, int me, int *myrange){
  // divide by number of nodes
  int chunksize = n / (NNodes -1);
  myrange[0] = (me-1) * chunksize;
  if(me < NNodes-1) myrange[1] = (me) * chunksize - 1;
  else myrange[1] = n - 1;
}

void init(){
  // find out the number of node using
  MPI_Comm_size(MPI_COMM_WORLD, &NNodes);
  // node ids
  MPI_Comm_rank(MPI_COMM_WORLD, &Me);
}

void findlinks(int* adjm,int* myrows,int n ,int *sum, int* num1sout, int* num1sin){
  for(int i = myrows[0]; i <= myrows[1]; i++){
    for(int j = 0; j < n; j++){
      if(adjm[i*n + j ] == 1){
        num1sout[*sum]= i;
        num1sin[*sum] = j;
        *sum = *sum +1;
        //printf("rows total %d %d %d: %d\n",i,j,n,*sum);
      }
    }
  }
}

void managerNode(int* adjm,int n){
  int myrows[2];

  // node 0 divides the work 
  // all other nodes do the work
  for(int i = 1; i < NNodes; i++){
    findmyrange(n,i,myrows);
    MPI_Send(myrows,2,
        MPI_INT, // type of data send
        i,// node which will be send to
        SENDROWS, // sending rows to work on
        MPI_COMM_WORLD);
    // MPI_Send(adjm,n*n,MPI_INT,i,SENDADJM,MPI_COMM_WORLD);
  }

}

int* transgraph(int* adjm,int n, int* nout){
  init();
  int* newmatrix = (int*)malloc(n*n*sizeof(int)); 
  int *num1sout = (int*)malloc(n*n*sizeof(int));
  int* num1sin = (int*)malloc(n*n*sizeof(int));

  if(Me == 0){
    managerNode(adjm,n);
  }
  else if(Me!= 0){
    MPI_Recv(myrows,2,MPI_INT,0,SENDROWS, MPI_COMM_WORLD,&status);
    int sum1 = 0;
    findlinks(adjm,myrows,n,&sum1,num1sout,num1sin);
    total[0] = sum1;

    MPI_Send(total,1,MPI_INT,
        0, // send to node 0
        SENDRESULT, // define total
        MPI_COMM_WORLD);

    MPI_Send(num1sin,n*n,MPI_INT,
        0, // send to node 0
        SENDRESULT1, // define total
        MPI_COMM_WORLD);

    MPI_Send(num1sout,n*n,MPI_INT,
        0, // send to node 0
        SENDRESULT2, // define total
        MPI_COMM_WORLD);
  }
  if(Me == 0){
    int sum = 0;
    int *intotal = (int*) calloc(n*n,sizeof(int)); 
    int *outotal = (int*)calloc(n*n,sizeof(int));
    int sum1;
    for(int i = 1; i < NNodes; i++){
      MPI_Recv(total,1,MPI_INT,i,SENDRESULT,MPI_COMM_WORLD,&status);
      MPI_Recv(num1sin,n*n,MPI_INT,i,SENDRESULT1,MPI_COMM_WORLD,&status);
      MPI_Recv(num1sout,n*n,MPI_INT,i,SENDRESULT2,MPI_COMM_WORLD,&status);

      for(int i = 0; i < total[0]; i++){
        intotal[i+sum] = num1sin[i];
        outotal[i+sum] = num1sout[i];
      }
      MPI_Get_count(&status,MPI_INT,&sum1);
      sum = sum + total[0];
    }

    int counter = 0;
    for(int i = 0; i < sum; i++){
      newmatrix[counter++] = outotal[i];
      newmatrix[counter++] = intotal[i];
    }
    counter = 0;

    printf("SUM: %d\n", sum);
    *nout = sum;
    //printf("count: %d\n",sum);
  }
  MPI_Bcast(newmatrix, n*n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(nout, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return newmatrix;
}
