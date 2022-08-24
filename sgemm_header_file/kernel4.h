#include <cstdio>

__global__ void sgemm_v4(float alpha, float *A, float *B, float beta, float *C, int N)
{

	const int tile_size = 32; 

	__shared__ float As[tile_size * tile_size]; 
    __shared__ float Bs[tile_size * tile_size]; 

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x, ty = threadIdx.y; 
    
    int row = tx * 4, col = ty;
	
	int abegin = tile_size * bx; 
	int astep = tile_size * N; 

	int bbegin = by * tile_size * N; 
	int bstep = tile_size; 
	int bend = bbegin + N;


    float Csub[4] = {0.0, 0.0, 0.0, 0.0}; 
    float b1;
 
    for(int a = abegin, b = bbegin; b < bend;  a += astep, b += bstep)
    {

        #pragma unroll
        for(int i=0;i<4;i++)
        {
            As[row + i + col * tile_size] = A[ a + N * col + row + i]; 
            Bs[row + i + col * tile_size] = B[b + N*col + row + i];
        }

        __syncthreads(); 
        
        #pragma unroll
        for(int k=0;k<tile_size;k++)
        {
        	b1 = Bs[k + col * tile_size]; 
            
            #pragma unroll 
            for(int i=0;i<4;i++)
            {
                 Csub[i] += As[row + i + k * tile_size] * b1;
            }
        }
     
        __syncthreads(); 
    }

    int id; 

    #pragma unroll 
    for(int i=0;i<4;i++)
    {
        id = abegin + bbegin + N * col + row + i;
        C[id] = Csub[i] * alpha + C[id]*beta;
    }

}
