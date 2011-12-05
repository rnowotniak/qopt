#include "mCEC_Function.h"
#include "CEC2011.h"
#pragma comment(lib, "CEC2011.lib")
#pragma comment(lib, "mclmcrrt.lib")
//---------------------------------------------------------------------------------------
int Initial_CEC2011_Cost_Function(void)
{
  if( !mclInitializeApplication(NULL,0) ) 
  { 
    std::cout << "Could not initialize the application!!!" << std::endl;
    return _HAS_ERROR; 
  } 

  // initialize lib
  if( !CEC2011Initialize())
  {
    std::cout << "Could not initialize Cost function!!!" << std::endl;
    return _HAS_ERROR; 
  }
  return _NO_ERROR;
}
//---------------------------------------------------------------------------------------
void Terminate_CEC2011_Cost_Function(void)
{
  CEC2011Terminate();
}
//---------------------------------------------------------------------------------------
void cost_function1(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 6, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 1;
  for(i=0;i<6;i++)
  {
    _x[i] = x[i];
  }
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function2(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 30, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 2;
  for(i=0;i<30;i++)
  {
    _x[i] = x[i];
  }
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function3(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 3;
  _x[0] = x[0];
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function4(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 4;
  _x[0] = x[0];
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function5(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 30, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 5;
  for(i=0;i<30;i++)
  {
    _x[i] = x[i];
  }
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function6(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 30, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 6;
  for(i=0;i<30;i++)
  {
    _x[i] = x[i];
  }
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function7(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 20, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 7;
  for(i=0;i<20;i++)
  {
    _x[i] = x[i];
  }
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function8(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[2] = {0,0};
  static mxArray *prhs[1] = {0};
  double *_x, *_fun_num,*_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 7, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _fun_num = mxGetPr(plhs[1]);
  
  *_fun_num = 8;
  for(i=0;i<7;i++)
  {
    _x[i] = x[i];
  }
  mlfBench_func(1, prhs, plhs[0], plhs[1]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function9(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[3] = {0,0,0};
  double *_x, *_f, *penaly, *rate_d;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 126, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[2] == 0)
    prhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<126;i++)
  {
    _x[i] = x[i];
  }
  mlfCost_fn(3, prhs, prhs+1, prhs+2, plhs[0]);
  _f = mxGetPr(prhs[0]);
  penaly = mxGetPr(prhs[1]);
  rate_d = mxGetPr(prhs[2]);
  f[0] = *_f;
  f[1] = *penaly;
  f[2] = *rate_d;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------

void cost_function10(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[4] = {0,0,0,0};
  static mxArray *prhs[3] = {0,0,0};
  double *_x, *_f, *sllreturn, *bwfn,*_null,*_phi_desired,*_distance;
  int i;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 12, mxREAL);
  if(plhs[1] == 0)
    plhs[1] = mxCreateDoubleMatrix(1, 2, mxREAL);
  if(plhs[2] == 0)
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(plhs[3] == 0)
    plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[2] == 0)
    prhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  _null = mxGetPr(plhs[1]);
  _phi_desired = mxGetPr(plhs[2]);
  _distance = mxGetPr(plhs[3]);
  //----------------------------
  _null[0] = 50;
  _null[1] = 120;
  _phi_desired[0] = 180;
  _distance[0] = 0.5;
  //----------------------------
  for(i=0;i<12;i++)
  {
    _x[i] = x[i];
  }
  mlfAntennafunccircular(3, prhs, prhs+1, prhs+2, plhs[0],plhs[1],plhs[2],plhs[3]);
  _f = mxGetPr(prhs[0]);
  sllreturn = mxGetPr(prhs[1]);
  bwfn = mxGetPr(prhs[2]);
  //std::cout<< "(m,n) = ( " << mxGetM(prhs[1]) <<" , "<< mxGetN(prhs[1]) <<" )"<<std::endl;
  //std::cout<< "(m,n) = ( " << mxGetM(prhs[2]) <<" , "<< mxGetN(prhs[2]) <<" )"<<std::endl;

  f[0] = *_f;
  f[1] = *sllreturn;
  f[2] = *bwfn;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function11_5(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;
  size_t n;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 120, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 10, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 10, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<120;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION11_5(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
  n = mxGetN(prhs[1]);
  for(i=1;i<n;i++)
  {
    f[i] = count[i-1];
  }
}
//---------------------------------------------------------------------------------------
void cost_function11_10(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;
  size_t n;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 240, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<240;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION11_10(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
  n = mxGetN(prhs[1]);
  for(i=1;i<n;i++)
  {
    f[i] = count[i-1];
  }
}
//---------------------------------------------------------------------------------------
void cost_function12_6(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 6, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<6;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION12_6(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
}
//---------------------------------------------------------------------------------------
void cost_function12_13(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 13, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<13;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION12_13(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
}
//---------------------------------------------------------------------------------------
void cost_function12_15(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 15, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<15;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION12_15(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
}
//---------------------------------------------------------------------------------------
void cost_function12_40(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 40, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<40;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION12_40(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
}
//---------------------------------------------------------------------------------------
void cost_function12_140(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 140, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<140;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION12_140(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
}


//---------------------------------------------------------------------------------------
void cost_function13_1(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;
  size_t n;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 96, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<96;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION13_1(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
  n = mxGetN(prhs[1]);
  for(i=1;i<n;i++)
  {
    f[i] = count[i-1];
  }
}
//---------------------------------------------------------------------------------------
void cost_function13_2(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;
  size_t n;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 96, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<96;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION13_2(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
  n = mxGetN(prhs[1]);
  for(i=1;i<n;i++)
  {
    f[i] = count[i-1];
  }
}
//---------------------------------------------------------------------------------------
void cost_function13_3(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[2] = {0,0};
  double *_x, *_f, *count;
  unsigned int i;
  size_t n;

  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 96, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  if(prhs[1] == 0)
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<96;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION13_3(2, prhs, prhs+1, plhs[0]);
  _f = mxGetPr(prhs[0]);
  count = mxGetPr(prhs[1]);
  f[0] = _f[0];
  n = mxGetN(prhs[1]);
  for(i=1;i<n;i++)
  {
    f[i] = count[i-1];
  }
}
//---------------------------------------------------------------------------------------
void cost_function14(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[1] = {0};
  double *_x, *_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 26, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<26;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION14(1, prhs, plhs[0]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
void cost_function15(FIELD_TYPE *x, FIELD_TYPE *f)
{
  static mxArray *plhs[1] = {0};
  static mxArray *prhs[1] = {0};
  double *_x,*_f;
  int i;
  if(plhs[0] == 0)
    plhs[0] = mxCreateDoubleMatrix(1, 22, mxREAL);
  if(prhs[0] == 0)
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  
  
  _x = mxGetPr(plhs[0]);
  
  for(i=0;i<22;i++)
  {
    _x[i] = x[i];
  }
  mlfMY_FUNCTION15(1, prhs, plhs[0]);
  _f = mxGetPr(prhs[0]);
  *f = *_f;
  //mxDestroyArray(temp);
}
//---------------------------------------------------------------------------------------
