#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>
#include "utils.h"
#include "cudamacro.h"
#include "kernel.cuh"
using namespace std; 

int m, n; 


void sgmmv_gpu(int kernel_num, int m, int n, float alpha, float *A, int LDA, float *x, float beta, float *y )
{

	if(kernel_num == 1)
	{
		int thread = 128; 
		int block = m/thread + (m%thread==0?0:1); 

		sgemv_kernel_v1<<<block, thread>>>(m, n, alpha, A, m, x, beta, y); 	
	}
	else if(kernel_num == 2)
	{
		int thread = 32;
		int dm = thread * 4;  
		int block = m/dm + (m%dm==0?0:1);

		sgemv_kernel_v2<<<block, thread>>>(m, n, alpha, A, m, x, beta, y); 	

	}

	else if(kernel_num == 3)
	{
		int thread = 32;
		int dm = thread * 4;  
		int block = m/dm + (m%dm==0?0:1);

		sgemv_kernel_v3<<<block, thread>>>(m, n, alpha, A, m, x, beta, y); 	

	}
	
	
}

int main(int argc, char* argv[])
{


	int kernel_num = 1; 
	if(argc == 2) kernel_num=atoi(argv[1]);


	int c = 10240; 
	m = c, n = c; 

	float *A, *x, *y, *ref; 
	float *dA, *dx, *dy;


	A = (float*)malloc( sizeof(float)*m*n ); 
	x = (float*)malloc( sizeof(float)*n ); 
	y = (float*)malloc( sizeof(float)*m ); 
	ref = (float*)malloc( sizeof(float)*m ); 

	float alpha = 1.0, beta = 0.0; 
	float ms; 

	random_initial_matrix(A, m*n); 
	random_initial_matrix(x, n); 
	random_initial_matrix(y, m); 
	random_initial_matrix(ref, m); 

	cudaEvent_t start, stop; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 

	CHECK_CUDA( cudaMalloc(&dA, sizeof(float)*m*n) ) ;
	CHECK_CUDA( cudaMalloc(&dx, sizeof(float)*n) ); 
	CHECK_CUDA( cudaMalloc(&dy, sizeof(float)*m) ); 


	CHECK_CUDA( cudaMemcpy(dA, A, sizeof(float)*m*n, cudaMemcpyHostToDevice) );
	CHECK_CUDA( cudaMemcpy(dx, x, sizeof(float)*n, cudaMemcpyHostToDevice) );
	CHECK_CUDA( cudaMemcpy(dy, y, sizeof(float)*m, cudaMemcpyHostToDevice) ); 

	

	cublasHandle_t sgemvhandle;  
	cublasCreate(&sgemvhandle); 

	cudaEventRecord(start, 0); 

	cublasSgemv(sgemvhandle, CUBLAS_OP_N,  m, n, &alpha, dA, m, dx, 1, &beta, dy, 1); 

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 

	cudaEventElapsedTime(&ms, start, stop); 

	cout<<"===================GPU CUBLAS code====================="<<endl;
	cout<<"Elapsed time: "<<ms<<"ms"<<endl; 
	printf("FLOPS: %f GFLOPS.\n\n", 2.0*1e-6*m*n/ms);

	CHECK_CUDA( cudaMemcpy(ref, dy, sizeof(float)*m, cudaMemcpyDeviceToHost) ); 

	cudaDeviceSynchronize(); 
	
	
	cudaEventRecord(start, 0); 
	sgmmv_gpu(kernel_num, m, n, alpha, dA, m, dx, beta, dy); 

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 

	cudaEventElapsedTime(&ms, start, stop); 


	cout<<"===================MY GPU code====================="<<endl;
	cout<<"Elapsed time: "<<ms<<"ms"<<endl; 
	printf("FLOPS: %f GFLOPS.\n\n", 2.0*1e-6*m*n/ms);



	
	CHECK_CUDA( cudaMemcpy(y, dy, sizeof(float)*m, cudaMemcpyDeviceToHost)); 

	cudaDeviceSynchronize(); 

	if(!compare_matrix(ref, y, m))
	{
		cout<<"Wrong kernel code!!"<<endl; 
	}
	else
	{
		cout<<"Right!"<<endl; 
	}


	free(A); free(x); free(y); free(ref); 
	cudaFree(dA); cudaFree(dx); cudaFree(dy); 





	return 0; 
}