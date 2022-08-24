#include <cstdio>




void random_initial_matrix(float * mat, int N)
{
	srand(time(NULL));
	for(int i=0;i<N*N;i++)
		mat[i] = (float)rand()/RAND_MAX;
}