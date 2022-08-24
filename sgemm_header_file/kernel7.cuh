#include <cstdio>

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}


__global__ void sgemm_v7(float alpha, float *A, float *B, float beta, float *C, int N)
{

	const int Ns = 128, Ks = 8; 

	__shared__ float As[Ns*Ks]; 
    __shared__ float Bs[Ns*Ks]; 

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x; 
    
    int rowa = (tx%32)*4 , cola = tx/32;
    int rowb = (tx%2)*4, colb = tx/2;
    // int rowc = (tx % 16)*8, colc = (tx/16)*8; 
	
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int rowc = (warp_row<<5) + (row_w<<3), colc = (warp_col<<6) + (col_w<<3);


	int abegin = Ns * bx; 
	int astep = Ks * N; 

	int bbegin = by * Ns * N; 
	int bstep = Ks; 
	int bend = bbegin + N;


    // float Csub[8][8] = {0.0}; 
    // float a1[8] = {0.0};
    float b1[8] = {0.0}; 
    float4 Av1, Av2, Bv1;
    // float4 Bv2;  
    
    float4 Csum[16]; 
    memset(Csum, 0, sizeof(Csum)); 

    float4 Cv[16]; 

 
    for(int a = abegin, b = bbegin; b < bend;  a += astep, b += bstep)
    {

        //load matrix A into shared memory
        
        Av1 = *( (float4 *)(&A[a + N * cola + rowa]) );
        *((float4 *)(&As[rowa + cola * Ns])) = Av1;


        //load matrix B into shared memory
        Bv1 = *( (float4 *)(&B[b + N*colb + rowb]) );
        // *((float4 *)(&Bs[rowb + colb * Ks])) = Bv1;

        Bs[colb + rowb * Ns] = Bv1.x; 
        Bs[colb + (rowb + 1) * Ns] = Bv1.y; 
        Bs[colb + (rowb + 2) * Ns] = Bv1.z; 
        Bs[colb + (rowb + 3) * Ns] = Bv1.w;

        __syncthreads(); 
        
        #pragma unroll
        for(int k=0;k<Ks;k++)
        {

            // load matrix A into register from shared memory
        	// #pragma unroll 
         //    for(int i=0;i<8;i++)
         //        a1[i] = As[rowc + i + k * Ns];
            Av1 = *( (float4 *)(&As[rowc + k*Ns]) );
            Av2 = *( (float4 *)(&As[rowc + 4 + k*Ns]) );

            
            // load matrix B into register from shared memory
            #pragma unroll 
            for(int i=0;i<8;i++)
                b1[i] = Bs[colc + i + k*Ns]; 

            #pragma unroll 
            for(int i=0;i<8;i++)
            {
                Csum[i*2 ] += Av1 * b1[i]; 
                Csum[i*2 + 1] += Av2 * b1[i]; 
            }


            
            /*

            // load matrix B into register from shared memory
            Bv1 = *( (float4 *) (&Bs[colc + k*Ns]) ); 
            Bv2 = *( (float4 *)(&Bs[colc + 4 + k*Ns]) ); 

            Csum[0] += Av1 * Bv1.x; 
            Csum[1] += Av2 * Bv1.x;
            Csum[2] += Av1 * Bv1.y; 
            Csum[3] += Av2 * Bv1.y;
            Csum[4] += Av1 * Bv1.z; 
            Csum[5] += Av2 * Bv1.z;
            Csum[6] += Av1 * Bv1.w; 
            Csum[7] += Av2 * Bv1.w;

            Csum[8] += Av1 * Bv2.x; 
            Csum[9] += Av2 * Bv2.x;
            Csum[10] += Av1 * Bv2.y; 
            Csum[11] += Av2 * Bv2.y;
            Csum[12] += Av1 * Bv2.z; 
            Csum[13] += Av2 * Bv2.z;
            Csum[14] += Av1 * Bv2.w; 
            Csum[15] += Av2 * Bv2.w;         
            */
        }
     
        __syncthreads(); 
    }

    
    #pragma unroll 
    for(int i=0;i<8;i++)
    {
        Cv[i*2] = *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc ]) );
        Cv[i*2+1] = *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc +4]) );
    }


    #pragma unroll 
    for(int i=0;i<16;i++)
    {
        Cv[i] = Csum[i] * alpha + Cv[i] * beta; 
    }


    #pragma unroll 
    for(int i=0;i<8;i++)
    {
        *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc ]) ) = Cv[i*2];
        *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc +4]) ) = Cv[i*2+1];
    }

  
}
