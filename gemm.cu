#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// #include <omp.h>
#include "kernel.cuh"
using namespace std;


void test_sgemm_kernel(int num, int N, float alpha, float beta, float *dA, float *dB, float *dC)
{
    
    if(num == 0)
    {
        dim3 blocks(N/32, N/32); 
        dim3 threads(32, 32);
        sgemm_v0<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 1)
    {
        dim3 blocks(N/32, N/32); 
        dim3 threads(32, 32);
        sgemm_v1<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 2)
    {
        dim3 blocks(N/32, N/32); 
        dim3 threads(32, 32);
        sgemm_v2<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 3)
    {
        dim3 blocks(N/32, N/32); 
        dim3 threads(8, 32);
        sgemm_v3<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 4)
    {
        dim3 blocks(N/32, N/32); 
        dim3 threads(8, 32);
        sgemm_v4<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 5)
    {
        dim3 blocks(N/64, N/64); 
        dim3 threads(16, 16);

        sgemm_v5<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 6)
    {
        dim3 blocks(N/128, N/128); 
        int threads = 256;
        sgemm_v6<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 7)
    {
        dim3 blocks(N/128, N/128); 
        int threads = 256;
        sgemm_v7<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }
    else if(num == 8)
    {
        dim3 blocks(N/128, N/128); 
        int threads = 256;
        sgemm_v8<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
    }

    // cudaDeviceSynchronize();


}




int main(int argc, char* argv[])
 {
    
    int kernel_num = 8;
    int sys_size = 2048; 

    char ch; 
    while((ch = getopt(argc, argv, "k:n:")) != EOF)
    {
        switch(ch)
        {
            case 'k' : kernel_num = atoi(optarg);
            break; 
            case 'n' : sys_size = atoi(optarg); 
            break; 

        }
    }

    if(kernel_num > 8 or kernel_num < 0)
        kernel_num = 8; 

    int N = sys_size; 

    printf("\nMatrix Size: %d X %d\n", N, N ); 
    cout<<"Kernel number: "<<kernel_num<<endl<<endl;

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

    cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC_ref, N);


    cudaEventRecord(start, 0);

    for(int i=0;i<10;i++)
        cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC_ref, N);

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop); 

    ms /= 10.0; 
  
    cout<<"CUBLAS Time Elapsed: "<<ms<<"ms"<<endl; 
    printf("FLOPS： %.3f GFLOPS\n\n", 2.*1e-6*N*N*N/ms);

    cudaMemcpy(C_ref, dC_ref, numbyte, cudaMemcpyDeviceToHost); 



    cout<<"===================MY GPU code====================="<<endl;

    
    
    cudaEventRecord(start, 0); 
    
    dim3 blocks(16, 16); 
    int threads = 256;
    // dim3 threads(16, 16);
    // test_sgemm_kernel(kernel_num, N, alpha, beta, dA, dB, dC);
    sgemm_v8<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);

    // mysgemm_v11<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC);
    // matrixmul<<<blocks, threads>>>(1.0, dA, dB, 0.0, dC, N); 

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&ms, start, stop); 
  
    cout<<"Time Elapsed: "<<ms<<"ms"<<endl; 

    printf("FLOPS： %.3f GFLOPS\n\n", 2.*1e-6*N*N*N/ms);
  
    cudaMemcpy(gC, dC, numbyte, cudaMemcpyDeviceToHost); 




    

    // cout<<gC[0]<<endl;
    // cout<<C_ref[0]<<endl;  
    
    float error = 0.0; 
    for(int i=0;i<N*N;i++)
        error = max(error, fabs(C_ref[i] - gC[i])); 
    
    printf("Max error is %.5f\n", error); 
    // cout<<"Max error is: "<<error<<endl; 


    free(A); free(B); free(C_ref); free(gC); 
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);




     return 0; 
 }
