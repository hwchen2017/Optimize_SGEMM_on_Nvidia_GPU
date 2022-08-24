#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "utils.h"
using namespace std; 

void random_initial_matrix(float* A, int m)
{

	srand(time(NULL)); 

	for(int i=0;i<m;i++)
		A[i] = (rand()/(double)RAND_MAX -0.5) ; 
}


bool compare_matrix(float *C, float *C_ref, int m)
{

	for(int i=0;i<m;i++)
		if(fabs(C_ref[i] - C[i]) >= 1e-3)
		{
			cout<<i<<"th element: "<<C_ref[i]<<"  "<<C[i]<<endl; 
			return false; 
		}

	return true; 
}
