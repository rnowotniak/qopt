/*
  CEC13 Test function suite 
  Jane Jing Liang (email: liangjing@zzu.edu.cn) 
  Dec. 23th 2012
*/

// #include <WINDOWS.H>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>


void test_func(double *, double *,int,int,int);

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag;


int main()
{
	int i,j,k,n,m,func_num;
	double *f,*x;
	FILE *fpt;

	m=2;
	n=10;

		fpt=fopen("input_data/shift_data.txt","r");
		if (fpt==NULL)
		{
			printf("\n Error: Cannot open input file for reading \n");
		}
		x=(double *)malloc(m*n*sizeof(double));
		if (x==NULL)
			printf("\nError: there is insufficient memory available!\n");
		for(i=0;i<n;i++)
		{
				fscanf(fpt,"%lf",&x[i]);
				printf("%f\n",x[i]);
		}
		fclose(fpt);

		for (i = 1; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
				x[i*n+j]=0.0;
				printf("%f\n",x[i*n+j]);
			}
		}


	f=(double *)malloc(sizeof(double)  *  m);
	for (i = 0; i < 28; i++)
	{
		func_num=i+1;
		for (k = 0; k < 1; k++)
		{
			test_func(x, f, n,m,func_num);
			for (j = 0; j < m; j++)
				printf(" f%d(x[%d]) = %f,",func_num,j+1,f[j]);
			printf("\n");
		}
	}
	free(x);
	free(f);
	free(y);
	free(z);
	free(M);
	free(OShift);
	free(x_bound);
}


