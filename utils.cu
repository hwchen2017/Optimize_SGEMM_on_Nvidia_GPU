#include <cstdio>




void random_initial_matrix(float * mat, int N)
{
	srand(time(NULL));
	for(int i=0;i<N*N;i++)
		mat[i] = (float)rand()/RAND_MAX;
}

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

/*
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
*/