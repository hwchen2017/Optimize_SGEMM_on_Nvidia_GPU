#include <cstdio>

__global__ void sgemm_v3(float alpha, float *A, float *B, float beta, float *C, int N)
{

	const int tile_size = 32; 

	__shared__ float As[tile_size * tile_size]; 
    __shared__ float Bs[tile_size * tile_size]; 

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x, ty = threadIdx.y; 
    
    int row1 = tx * 4, col = ty;
    int row2 = row1 + 1, row3 = row1 + 2, row4 = row1 + 3; 
	
	int abegin = tile_size * bx; 
	int astep = tile_size * N; 

	int bbegin = by * tile_size * N; 
	int bstep = tile_size; 
	int bend = bbegin + N;


    float Csub[4] = {0.0, 0.0, 0.0, 0.0}; 
    float b1;
 
    for(int a = abegin, b = bbegin; b < bend;  a += astep, b += bstep)
    {
        As[row1 + col * tile_size] = A[ a + N * col + row1];
        As[row2 + col * tile_size] = A[ a + N * col + row2];
        As[row3 + col * tile_size] = A[ a + N * col + row3];
        As[row4 + col * tile_size] = A[ a + N * col + row4];

        Bs[row1 + col * tile_size] = B[b + N*col + row1];
        Bs[row2 + col * tile_size] = B[b + N*col + row2];
        Bs[row3 + col * tile_size] = B[b + N*col + row3];
        Bs[row4 + col * tile_size] = B[b + N*col + row4];

        __syncthreads(); 
        
        #pragma unroll
        for(int k=0;k<tile_size;k++)
        {
        	b1 = Bs[k + col * tile_size]; 

            Csub[0] += As[row1 + k * tile_size] * b1;
            Csub[1] += As[row2 + k * tile_size] * b1;
            Csub[2] += As[row3 + k * tile_size] * b1;
            Csub[3] += As[row4 + k * tile_size] * b1;
        }
     
        __syncthreads(); 
    }

    int id0 = abegin + bbegin + N * col + row1;
    int id1 = abegin + bbegin + N * col + row2;
    int id2 = abegin + bbegin + N * col + row3;
    int id3 = abegin + bbegin + N * col + row4;
    
    C[id0] = Csub[0] * alpha + C[id0]*beta;
    C[id1] = Csub[1] * alpha + C[id1]*beta;
    C[id2] = Csub[2] * alpha + C[id2]*beta;
    C[id3] = Csub[3] * alpha + C[id3]*beta;

}
