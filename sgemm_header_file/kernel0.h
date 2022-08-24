#include <cstdio>

__global__ void sgemm_v0(float alpha, float *A, float *B, float beta, float *C, int N)
{

	int bx = blockIdx.x, by = blockIdx.y; 
	int tx = threadIdx.x, ty = threadIdx.y; 

	int i = bx * blockDim.x + tx; 
	int j = by * blockDim.y + ty; 

	C[i + j*N]  = beta * C[i + j*N];


	for(int k=0;k<N;k++)
	{
		C[i + j*N] = C[i + j*N] + alpha * A[i + N*k] * B[ k + j*N];
	}


}