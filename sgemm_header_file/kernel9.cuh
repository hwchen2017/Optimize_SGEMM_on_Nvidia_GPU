#include <cstdio>

__global__ void sgemm_v9(float alpha, float *A, float *B, float beta, float *C, int N)
{

	const int Ns = 128, Ks = 8; 

	__shared__ float As[2][Ns*Ks]; 
    __shared__ float Bs[2][Ns*Ks]; 

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x; 
    
    int rowa = (tx%32)*4, cola = tx/32;
    int rowb = (tx%2)*4, colb = tx/2;
	
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int rowc = (warp_row<<5) + (row_w<<3), colc = (warp_col<<6) + (col_w<<3);


	int abegin = Ns * bx; 

	int bbegin = by * Ns * N; 


    //for computatiom
    // float B_reg[2][8];
    float4 B_reg1[2], B_reg2[2];
    float4 A_reg1[2], A_reg2[2]; 

    float4 A_nxtile, B_nxtile; 

    
    float4 Cv[16]; 
    float4 Csum[16]; 
    memset(Csum, 0, sizeof(Csum)); 

    int write_id = 1; 


    //load first tile to shared memory
    int a = abegin, b = bbegin; 
    //load tile in A
    A_nxtile = *( (float4 *)(&A[a + N * cola + rowa]) );

    *((float4 *)(&As[0][rowa + cola * Ns])) = A_nxtile;

    //load tile in B
    B_nxtile = *( (float4 *)(&B[b + N*colb + rowb]) );

    Bs[0][colb + rowb * Ns] = B_nxtile.x; 
    Bs[0][colb + (rowb + 1) * Ns] = B_nxtile.y; 
    Bs[0][colb + (rowb + 2) * Ns] = B_nxtile.z; 
    Bs[0][colb + (rowb + 3) * Ns] = B_nxtile.w;

    __syncthreads(); 


    // load from shared memory to register
    // load matrix B into register
    A_reg1[0] = *( (float4 *)(&As[0][rowc + 0*Ns]) );
    A_reg2[0] = *( (float4 *)(&As[0][rowc + 4 + 0*Ns]) );

    // load matrix B into register

    B_reg1[0] = *( (float4 *)(&Bs[0][colc + 0*Ns]) );
    B_reg2[0] = *( (float4 *)(&Bs[0][colc + 4 + 0*Ns]) );


    int inc;    

 
    for(int kc=0; kc < N;  kc+=Ks)
    {

        inc = (kc + Ks) % N; 

        a = abegin + inc * N;
        b = bbegin + inc;


        A_nxtile = *( (float4 *)(&A[a + N * cola + rowa]) );    

        B_nxtile = *( (float4 *)(&B[b + N*colb + rowb]) );

        
        int load_id = 1 - write_id; 
        int next_id, cur_id, nxtk; 

        #pragma unroll
        for(int k=0;k<Ks;k++)
        {

            cur_id = k%2; 
            next_id = (k+1)%2; 
            nxtk = (k+1)%Ks;

            //load matrix A into register from shared memory
            A_reg1[next_id] = *( (float4 *)(&As[load_id][rowc + nxtk*Ns]) );
            A_reg2[next_id] = *( (float4 *)(&As[load_id][rowc + 4 + nxtk*Ns]) );

            // load matrix B into register from shared memory
            B_reg1[next_id] = *( (float4 *)(&Bs[load_id][colc + 0 + nxtk*Ns]) ); 
            B_reg2[next_id] = *( (float4 *)(&Bs[load_id][colc + 4 + nxtk*Ns]) );  


            //compute C element

            Csum[0 ] += A_reg1[cur_id] * B_reg1[cur_id].x;
            Csum[1] += A_reg2[cur_id] * B_reg1[cur_id].x; 
            Csum[2 ] += A_reg1[cur_id] * B_reg1[cur_id].y;
            Csum[3] += A_reg2[cur_id] * B_reg1[cur_id].y; 
            Csum[4 ] += A_reg1[cur_id] * B_reg1[cur_id].z;
            Csum[5] += A_reg2[cur_id] * B_reg1[cur_id].z; 
            Csum[6 ] += A_reg1[cur_id] * B_reg1[cur_id].w;
            Csum[7] += A_reg2[cur_id] * B_reg1[cur_id].w; 
            Csum[8 ] += A_reg1[cur_id] * B_reg2[cur_id].x;
            Csum[9] += A_reg2[cur_id] * B_reg2[cur_id].x; 
            Csum[10 ] += A_reg1[cur_id] * B_reg2[cur_id].y;
            Csum[11] += A_reg2[cur_id] * B_reg2[cur_id].y; 
            Csum[12 ] += A_reg1[cur_id] * B_reg2[cur_id].z;
            Csum[13] += A_reg2[cur_id] * B_reg2[cur_id].z; 
            Csum[14 ] += A_reg1[cur_id] * B_reg2[cur_id].w;
            Csum[15] += A_reg2[cur_id] * B_reg2[cur_id].w; 

        }


       *((float4 *)(&As[write_id][rowa + cola * Ns])) = A_nxtile; 
        Bs[write_id][colb + rowb * Ns] = B_nxtile.x; 
        Bs[write_id][colb + (rowb + 1) * Ns] = B_nxtile.y; 
        Bs[write_id][colb + (rowb + 2) * Ns] = B_nxtile.z; 
        Bs[write_id][colb + (rowb + 3) * Ns] = B_nxtile.w;

     
        __syncthreads();
 

        A_reg1[0] = *( (float4 *)(&As[write_id][rowc]) );
        A_reg2[0] = *( (float4 *)(&As[write_id][rowc + 4]) );

        B_reg1[0] = *( (float4 *)(&Bs[write_id][colc]) );
        B_reg2[0] = *( (float4 *)(&Bs[write_id][colc+4]) );

        write_id = 1 - write_id;

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
