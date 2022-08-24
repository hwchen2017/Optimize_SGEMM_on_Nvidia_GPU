#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// #include <omp.h>
#include "kernel.cuh"
using namespace std;


#define CUDA_KERNEL_CALLER(...) do{\
  if(cudaPeekAtLastError() != cudaSuccess){\
    printf("A CUDA error occurred prior to the kernel call %s at line %d\n", #__VA_ARGS__,  __LINE__); exit(1);\
  }\
  __VA_ARGS__;\
  cudaError_t cuda_ret = cudaPeekAtLastError();\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the kernel call %s\n", #__VA_ARGS__);\
    exit(1);\
  }\
}while(0)



__global__ void matrixmul(float alpha, float *A, float *B, float beta, float *C, int N)
{

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x, ty = threadIdx.y; 
    
    const int tile_size = 16; 
    
    __shared__ float As[tile_size][tile_size]; 
    __shared__ float Bs[tile_size][tile_size]; 
 
    int abegin = by * tile_size * N; 
    int aend = abegin + N -1;

    int bbegin = tile_size * bx; 
    int bstep = tile_size * N; 
    
    float Csub = 0.0; 
 
    for(int a = abegin, b = bbegin; a<=aend;  a += tile_size, b += bstep)
    {
        As[ty][tx] = A[ a + N * ty + tx]; 
        //Bs[tx][ty] = B[b + N*tx + ty ]; 
        Bs[ty][tx] = B[b + N*ty + tx];

        __syncthreads(); 
           
        for(int k=0;k<tile_size;k++)
        {
            Csub += As[ty][k] * Bs[k][tx]; 
           // Csub += As[ty][k] * Bs[tx][k];
        }
     
        __syncthreads(); 
    }
 
    // int c = abegin + bbegin + N * ty + tx;
    // C[c] = Csub; 

    int id = abegin + bbegin + N * ty + tx;
    C[id] = Csub * alpha + C[id]*beta;

}




 int main()
 {
    
    // cout<<"hello"<<endl; 

    int N = 2048; 

    float *A, *B, *C_ref, *gC; 
    int numbyte = sizeof(float) * N*N; 
    
    A = (float *)malloc(numbyte); 
    B = (float *)malloc(numbyte); 
    C_ref = (float *)malloc(numbyte); 
    gC = (float *)malloc(numbyte); 

    float alpha = 1.0f, beta = 0.0f;
    
    srand(time(NULL)); 
  
    for(int i=0;i<N*N;i++)
    {
        A[i] = (double)rand()/RAND_MAX;
        B[i] = (double)rand()/RAND_MAX;  
    }

    

    // cpu parallel code 

    /*
    cout<<"===================CPU code====================="<<endl;

    cout<<"Threads: "<<omp_get_max_threads()<<endl; 

    double st = omp_get_wtime();

    #pragma omp parallel for
      for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
        {
            C[i+j*N] = 0.0; 
            for(int k=0;k<N;k++)
                C[i+j*N] += A[i+k*N]*B[k+j*N]; 
        }

    // cout<<C[0]<<endl; 
    
    cout<<"CPU Time: "<<omp_get_wtime()-st<<"s"<<endl; 

    */
    
    
    float *dA, *dB, *dC, *dC_ref; 
    
    cudaMalloc(&dA, numbyte); 
    cudaMalloc(&dB, numbyte); 
    cudaMalloc(&dC, numbyte); 
    cudaMalloc(&dC_ref, numbyte); 

    
    cudaMemcpy(dA, A, numbyte, cudaMemcpyHostToDevice); 
    cudaMemcpy(dB, B, numbyte, cudaMemcpyHostToDevice);

    float ms; 
  
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 



    cout<<"===================GPU CUBLAS code====================="<<endl;

    
    cublasHandle_t err; 
    cublasCreate(&err); 

    cudaEventRecord(start, 0);

    cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC_ref, N);

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop); 
  
    cout<<"CUBLAS Time Elapsed: "<<ms<<"ms"<<endl<<endl; 

    cudaMemcpy(C_ref, dC_ref, numbyte, cudaMemcpyDeviceToHost); 



    cout<<"===================MY GPU code====================="<<endl;

    
    
    cudaEventRecord(start, 0); 
    
    dim3 blocks(16, 16); 
    int threads = 256;
    // dim3 threads(16, 16);

    sgemm_v8<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);

    // mysgemm_v11<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC);
    // matrixmul<<<blocks, threads>>>(1.0, dA, dB, 0.0, dC, N); 

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop); 
  
    cout<<"Time Elapsed: "<<ms<<"ms"<<endl<<endl; 

    printf("FLOPS： %.3f GFLOPS\n", 2.*1e-6*N*N*N/ms);
  
    cudaMemcpy(gC, dC, numbyte, cudaMemcpyDeviceToHost); 




    

    // cout<<gC[0]<<endl;
    // cout<<C_ref[0]<<endl;  
    
    float error = 0.0; 
    for(int i=0;i<N*N;i++)
        error = max(error, fabs(C_ref[i] - gC[i])); 
    
    cout<<"Max error is: "<<error<<endl; 


    free(A); free(B); free(C_ref); free(gC); 
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);




     return 0; 
 }
