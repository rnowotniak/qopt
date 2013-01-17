/*
  CEC13 Test Function Suite 
  Jane Jing Liang (email: liangjing@zzu.edu.cn) 
  8th Jan. 2013
*/

test_func.cpp is the test function
Example:
test_func(x, f, dimension,population_size,func_num);

main.cpp is an example function about how to use test_func.cpp


#include <WINDOWS.H>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>
void test_func(double *, double *,int,int,int);
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag;
void main()
{
...
}