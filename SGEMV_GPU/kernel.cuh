#include <stdio.h>
#include <stdlib.h>


inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}


#define A(i, j) A[(i) + (j)*LDA]


__global__ void sgemv_kernel_v1(int m, int n, float alpha, float *A, int LDA,  float *x, float beta, float *y)
{

	int tx = threadIdx.x; 
	int bx = blockIdx.x, bm = blockDim.x; 

	if(bx*bm+tx >= m) return; 

	float ssum = 0.0; 

	for(int i=0;i<n;i++)
	{
		ssum += A(bx*bm+tx, i) * x[i]; 
	}

	y[bx*bm+tx] = alpha * ssum + beta * y[bx*bm+tx]; 

}


__global__ void sgemv_kernel_v2(int m, int n, float alpha, float *A, int LDA,  float *x, float beta, float *y)
{

	int tx = threadIdx.x * 4; 
	int bx = blockIdx.x, bm = blockDim.x * 4; 

	int tx1 = tx+1, tx2 = tx+2, tx3 = tx+3; 
	// if(bx*bm+tx >= m) return; 
	float xi; 
	float ssum[4];
	memset(ssum, 0, sizeof(ssum));  

	// #pragma unroll
	for(int i=0;i<n;i++)
	{
		xi = x[i]; 

		ssum[0] += A(bx*bm+tx, i) * xi;
		ssum[1] += A(bx*bm+tx1, i) * xi; 
		ssum[2] += A(bx*bm+tx2, i) * xi; 
		ssum[3] += A(bx*bm+tx3, i) * xi;  
	}

	y[bx*bm+tx] = alpha * ssum[0] + beta * y[bx*bm+tx]; 
	y[bx*bm+tx1] = alpha * ssum[1] + beta * y[bx*bm+tx1]; 
	y[bx*bm+tx2] = alpha * ssum[2] + beta * y[bx*bm+tx2]; 
	y[bx*bm+tx3] = alpha * ssum[3] + beta * y[bx*bm+tx3]; 


}



__global__ void sgemv_kernel_v3(int m, int n, float alpha, float *A, int LDA,  float *x, float beta, float *y)
{

	int tx = threadIdx.x * 4; 
	int bx = blockIdx.x, bm = blockDim.x * 4; 

	float4 ssum, va, vx, vy; 


	ssum = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 
	// #pragma unroll
	for(int i=0;i<n;i++)
	{
		
		vx = make_float4(x[i], x[i], x[i], x[i]);  

		va = *(float4 *)(&A(bx*bm+tx, i));   

		ssum += (va * vx); 
	}

	vy = *(float4 *)(&y[bx*bm + tx]); 
	
	vy = vy * beta + ssum * alpha; 

	*(float4 *)(&y[bx*bm + tx]) = vy; 	


}
