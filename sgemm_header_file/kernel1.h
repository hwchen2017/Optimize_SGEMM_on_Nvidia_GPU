#include <cstdio>

__global__ void sgemm_v1(float alpha, float *A, float *B, float beta, float *C, int N)
{

	int bx = blockIdx.x, by = blockIdx.y; 
	int tx = threadIdx.x, ty = threadIdx.y; 

	int i = bx * blockDim.x + tx; 
	int j = by * blockDim.y + ty; 

	float ssum = 0.0; 

	for(int k=0;k<N;k++)
	{
		ssum = ssum + A[i + N*k] * B[ k + j*N];
	}

	C[i + j*N] = alpha * ssum + beta * C[i + j*N]; 


}



