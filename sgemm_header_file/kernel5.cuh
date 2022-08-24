#include <cstdio>

__global__ void sgemm_v5(float alpha, float *A, float *B, float beta, float *C, int N)
{

	const int Ns = 64, Ks = 16; 

	__shared__ float As[Ns*Ks]; 
    __shared__ float Bs[Ns*Ks]; 

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x, ty = threadIdx.y; 
    
    int rowa = tx * 4, cola = ty;
    int rowb = tx, colb = 4*ty;
	
	int abegin = Ns * bx; 
	int astep = Ks * N; 

	int bbegin = by * Ns * N; 
	int bstep = Ks; 
	int bend = bbegin + N;


    float Csub[4][4] = {0.0}; 
    float a1[4] = {0.0};
    float b1[4] = {0.0}; 
 
    for(int a = abegin, b = bbegin; b < bend;  a += astep, b += bstep)
    {

        //load matrix A into shared memory
        #pragma unroll
        for(int i=0;i<4;i++)
        {
            As[rowa + i + cola * Ns] = A[ a + N * cola + rowa + i]; 
        }


        //load matrix B into shared memory
        #pragma unroll
        for(int i=0;i<4;i++)
        {
            Bs[rowb + (colb + i) * Ks] = B[b + N*(colb+i) + rowb];
        }

        __syncthreads(); 
        
        #pragma unroll
        for(int k=0;k<Ks;k++)
        {

            // load matrix A into register from shared memory
        	#pragma unroll 
            for(int i=0;i<4;i++)
                a1[i] = As[rowa + i + k * Ns]; 

            // load matrix B into register from shared memory
            #pragma unroll 
            for(int i=0;i<4;i++)
                b1[i] = Bs[k + (colb + i) * Ks]; 
            
            // Calculate C
            #pragma unroll
            for(int i=0;i<4;i++)
            {
                #pragma unroll 
                for(int j=0;j<4;j++)
                {
                    Csub[i][j] += a1[i] * b1[j]; 
                }
            }
            

        }
     
        __syncthreads(); 
    }



    // store C back
    int id[4][4]; 

    #pragma unroll 
    for(int j=0;j<4;j++)
    {
        #pragma unroll 
        for(int i=0;i<4;i++)
        {
            id[i][j] = abegin + bbegin + N * (colb + j) + rowa + i;
            C[ id[i][j] ] = Csub[i][j] * alpha + C[ id[i][j] ]*beta;
        }
        
    }

  
}
