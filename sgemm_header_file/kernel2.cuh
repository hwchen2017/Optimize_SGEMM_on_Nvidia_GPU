#include <cstdio>

__global__ void sgemm_v2(float alpha, float *A, float *B, float beta, float *C, int N)
{

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x, ty = threadIdx.y; 
    
    const int tile_size = 32; 
    
    __shared__ float As[tile_size * tile_size]; 
    __shared__ float Bs[tile_size * tile_size]; 
	
	int abegin = tile_size * bx; 
	int astep = tile_size * N;

	int bbegin = by * tile_size * N; 
	int bstep = tile_size; 
	int bend = bbegin + N;


    float Csub = 0.0; 
 
    for(int a = abegin, b = bbegin; b < bend;  a += astep, b += bstep)
    {

        As[tx + ty * tile_size] = A[ a + N * ty + tx]; 
        Bs[tx + ty * tile_size] = B[b + N*ty + tx];
         
        __syncthreads(); 
        
        #pragma unroll
        for(int k=0;k<tile_size;k++)
        {
            Csub += Bs[k + ty * tile_size] * As[tx + k * tile_size]; 
        }
     
        __syncthreads(); 
    }
 
    int id = abegin + bbegin + N * ty + tx;
    C[id] = Csub * alpha + C[id]*beta;

}