#include <cstdio>

__global__ void sgemm_v6(float alpha, float *A, float *B, float beta, float *C, int N)
{

	const int Ns = 128, Ks = 8; 

	__shared__ float As[Ns*Ks]; 
    __shared__ float Bs[Ns*Ks]; 

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x; 
    
    int rowa = (tx%32)*4 , cola = tx/32;
    int rowb = (tx%2)*4, colb = tx/2;
    int rowc = (tx % 16)*8, colc = (tx/16)*8; 
	
	int abegin = Ns * bx; 
	int astep = Ks * N; 

	int bbegin = by * Ns * N; 
	int bstep = Ks; 
	int bend = bbegin + N;


    float Csub[8][8] = {0.0}; 
    float a1[8] = {0.0};
    float b1[8] = {0.0}; 
    float4 Av1, Bv1; 
    // float4 Csub[4][4]; 
    // memset(Csub, 0, sizeof(Csub)); 

    // float4 test = make_float4(1.0, 2.0, 3.0, 4.0); 
 
    for(int a = abegin, b = bbegin; b < bend;  a += astep, b += bstep)
    {

        //load matrix A into shared memory
        
        Av1 = *( (float4 *)(&A[a + N * cola + rowa]) );
        *((float4 *)(&As[rowa + cola * Ns])) = Av1;

        // #pragma unroll
        // for(int i=0;i<4;i++)
        // {
        //     As[rowa + i + cola * Ns] = A[ a + N * cola + rowa + i]; 
        // }


        //load matrix B into shared memory
        Bv1 = *( (float4 *)(&B[b + N*colb + rowb]) );
        *((float4 *)(&Bs[rowb + colb * Ks])) = Bv1;

        // #pragma unroll
        // for(int i=0;i<4;i++)
        // {
        //     Bs[rowb + i + colb * Ks] = B[b + N*colb + rowb + i];
        // }

        __syncthreads(); 
        
        #pragma unroll
        for(int k=0;k<Ks;k++)
        {

            // load matrix A into register from shared memory
        	#pragma unroll 
            for(int i=0;i<8;i++)
                a1[i] = As[rowc + i + k * Ns];
            // Av1 = *( (float4 *)(&As[rowc + k*Ns]) );
            // Av2 = *( (float4 *)(&As[rowc + 4 + k*Ns]) )


            // load matrix B into register from shared memory
            #pragma unroll 
            for(int i=0;i<8;i++)
                b1[i] = Bs[k + (colc + i) * Ks]; 
            
            // Bv1 = make_float4(Bs[k + (colc + 0) * Ks], Bs[k + (colc + 1) * Ks], Bs[k + (colc + 2) * Ks], Bs[k + (colc + 3) * Ks] ); 
            // Bv2 = make_float4(Bs[k + (colc + 4) * Ks], Bs[k + (colc + 5) * Ks], Bs[k + (colc + 6) * Ks], Bs[k + (colc + 7) * Ks] );             


            // Calculate C
            #pragma unroll
            for(int i=0;i<8;i++)
            {
                #pragma unroll 
                for(int j=0;j<8;j++)
                {
                    Csub[i][j] += a1[i] * b1[j]; 
                }
            }
            

        }
     
        __syncthreads(); 
    }



    // store C back
    int id[8][8]; 

    #pragma unroll 
    for(int j=0;j<8;j++)
    {
        #pragma unroll 
        for(int i=0;i<8;i++)
        {
            id[i][j] = abegin + bbegin + N * (colc + j) + rowc + i;
            C[ id[i][j] ] = Csub[i][j] * alpha + C[ id[i][j] ]*beta;
        }
        
    }

  
}
