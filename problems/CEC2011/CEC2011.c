/*
 * MATLAB Compiler: 4.15 (R2011a)
 * Date: Tue Oct 30 02:39:33 2012
 * Arguments: "-B" "macro_default" "-W" "lib:CEC2011" "-T" "link:lib"
 * "matlab/angle3d.m" "matlab/antennafunccircular.m" "matlab/array_factorcir.m"
 * "matlab/bench_func.m" "matlab/bounds.m" "matlab/cassini2.m"
 * "matlab/cost_fn.m" "matlab/data6Bus.m" "matlab/diffsolv.m"
 * "matlab/display_plot.m" "matlab/EBEformybus.m" "matlab/EBEinputfile.m"
 * "matlab/fn_DED_10.m" "matlab/fn_DED_5.m" "matlab/fn_ELD_13.m"
 * "matlab/fn_ELD_140.m" "matlab/fn_ELD_15.m" "matlab/fn_ELD_40.m"
 * "matlab/fn_ELD_6.m" "matlab/fn_HT_ELD_Case_1.m" "matlab/fn_HT_ELD_Case_2.m"
 * "matlab/fn_HT_ELD_Case_3.m" "matlab/fourthh.m" "matlab/fourth.m"
 * "matlab/getlimit_cassini2.m" "matlab/getlimit_messenger.m" "matlab/hsdrf.m"
 * "matlab/intgrl1.m" "matlab/intgrl.m" "matlab/messengerfull.m"
 * "matlab/mga_dsm.m" "matlab/MY_FUNCTION11_10.m" "matlab/MY_FUNCTION11_5.m"
 * "matlab/MY_FUNCTION12_13.m" "matlab/MY_FUNCTION12_140.m"
 * "matlab/MY_FUNCTION12_15.m" "matlab/MY_FUNCTION12_40.m"
 * "matlab/MY_FUNCTION12_6.m" "matlab/MY_FUNCTION13_1.m"
 * "matlab/MY_FUNCTION13_2.m" "matlab/MY_FUNCTION13_3.m"
 * "matlab/MY_FUNCTION14.m" "matlab/MY_FUNCTION15.m" "matlab/tersoff.m"
 * "matlab/trapezoidalcir.m" "-d" "." 
 */

#include <stdio.h>
#define EXPORTING_CEC2011 1
#include "CEC2011.h"

static HMCRINSTANCE _mcr_inst = NULL;


#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultPrintHandler(const char *s)
{
  return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultErrorHandler(const char *s)
{
  int written = 0;
  size_t len = 0;
  len = strlen(s);
  written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
  if (len > 0 && s[ len-1 ] != '\n')
    written += mclWrite(2 /* stderr */, "\n", sizeof(char));
  return written;
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_CEC2011_C_API
#define LIB_CEC2011_C_API /* No special import/export declaration */
#endif

LIB_CEC2011_C_API 
bool MW_CALL_CONV CEC2011InitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
  if (_mcr_inst != NULL)
    return true;
  if (!mclmcrInitialize())
    return false;
    {
        mclCtfStream ctfStream = 
            mclGetEmbeddedCtfStream((void *)(CEC2011InitializeWithHandlers), 
                                    1884313);
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(   &_mcr_inst,
                                                                error_handler, 
                                                                print_handler,
                                                                ctfStream, 
                                                                1884313);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
  return true;
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV CEC2011Initialize(void)
{
  return CEC2011InitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}

LIB_CEC2011_C_API 
void MW_CALL_CONV CEC2011Terminate(void)
{
  if (_mcr_inst != NULL)
    mclTerminateInstance(&_mcr_inst);
}

LIB_CEC2011_C_API 
long MW_CALL_CONV CEC2011GetMcrID() 
{
  return mclGetID(_mcr_inst);
}

LIB_CEC2011_C_API 
void MW_CALL_CONV CEC2011PrintStackTrace(void) 
{
  char** stackTrace;
  int stackDepth = mclGetStackTrace(&stackTrace);
  int i;
  for(i=0; i<stackDepth; i++)
  {
    mclWrite(2 /* stderr */, stackTrace[i], sizeof(char)*strlen(stackTrace[i]));
    mclWrite(2 /* stderr */, "\n", sizeof(char)*strlen("\n"));
  }
  mclFreeStackTrace(&stackTrace, stackDepth);
}


LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxAngle3d(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "angle3d", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxAntennafunccircular(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                         *prhs[])
{
  return mclFeval(_mcr_inst, "antennafunccircular", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxArray_factorcir(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "array_factorcir", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxBench_func(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "bench_func", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxBounds(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "bounds", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxCassini2(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "cassini2", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxCost_fn(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "cost_fn", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxData6Bus(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "data6Bus", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxDiffsolv(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "diffsolv", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxDisplay_plot(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "display_plot", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxEBEformybus(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "EBEformybus", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxEBEinputfile(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "EBEinputfile", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_DED_10(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fn_DED_10", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_DED_5(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fn_DED_5", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_13(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fn_ELD_13", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_140(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fn_ELD_140", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_15(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fn_ELD_15", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_40(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fn_ELD_40", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_6(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fn_ELD_6", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_HT_ELD_Case_1(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[])
{
  return mclFeval(_mcr_inst, "fn_HT_ELD_Case_1", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_HT_ELD_Case_2(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[])
{
  return mclFeval(_mcr_inst, "fn_HT_ELD_Case_2", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_HT_ELD_Case_3(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[])
{
  return mclFeval(_mcr_inst, "fn_HT_ELD_Case_3", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFourthh(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fourthh", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFourth(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "fourth", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxGetlimit_cassini2(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                       *prhs[])
{
  return mclFeval(_mcr_inst, "getlimit_cassini2", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxGetlimit_messenger(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                        *prhs[])
{
  return mclFeval(_mcr_inst, "getlimit_messenger", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxHsdrf(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "hsdrf", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxIntgrl1(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "intgrl1", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxIntgrl(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "intgrl", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMessengerfull(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "messengerfull", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMga_dsm(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "mga_dsm", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION11_10(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION11_10", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION11_5(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION11_5", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_13(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION12_13", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_140(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                       *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION12_140", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_15(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION12_15", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_40(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION12_40", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_6(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION12_6", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION13_1(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION13_1", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION13_2(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION13_2", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION13_3(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION13_3", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION14(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION14", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION15(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "MY_FUNCTION15", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxTersoff(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "tersoff", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxTrapezoidalcir(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "trapezoidalcir", nlhs, plhs, nrhs, prhs);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfAngle3d(int nargout, mxArray** th, mxArray* x, mxArray* j, mxArray* 
                             i, mxArray* k)
{
  return mclMlfFeval(_mcr_inst, "angle3d", nargout, 1, 4, th, x, j, i, k);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfAntennafunccircular(int nargout, mxArray** y, mxArray** sllreturn, 
                                         mxArray** bwfn, mxArray* x1, mxArray* null, 
                                         mxArray* phi_desired, mxArray* distance)
{
  return mclMlfFeval(_mcr_inst, "antennafunccircular", nargout, 3, 4, y, sllreturn, bwfn, x1, null, phi_desired, distance);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfArray_factorcir(int nargout, mxArray** y, mxArray* x1, mxArray* phi, 
                                     mxArray* phi_desired, mxArray* distance, mxArray* 
                                     dim)
{
  return mclMlfFeval(_mcr_inst, "array_factorcir", nargout, 1, 5, y, x1, phi, phi_desired, distance, dim);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfBench_func(int nargout, mxArray** f, mxArray* x, mxArray* fun_num)
{
  return mclMlfFeval(_mcr_inst, "bench_func", nargout, 1, 2, f, x, fun_num);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfBounds()
{
  return mclMlfFeval(_mcr_inst, "bounds", 0, 0, 0);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfCassini2(int nargout, mxArray** J, mxArray* t, mxArray* problem)
{
  return mclMlfFeval(_mcr_inst, "cassini2", nargout, 1, 2, J, t, problem);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfCost_fn(int nargout, mxArray** fitness, mxArray** PENALTY, mxArray** 
                             rate_d, mxArray* x)
{
  return mclMlfFeval(_mcr_inst, "cost_fn", nargout, 3, 1, fitness, PENALTY, rate_d, x);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfData6Bus()
{
  return mclMlfFeval(_mcr_inst, "data6Bus", 0, 0, 0);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfDiffsolv(int nargout, mxArray** dy, mxArray* t, mxArray* x, mxArray* 
                              u)
{
  return mclMlfFeval(_mcr_inst, "diffsolv", nargout, 1, 3, dy, t, x, u);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfDisplay_plot(mxArray* gbest, mxArray* phi_desired, mxArray* distance)
{
  return mclMlfFeval(_mcr_inst, "display_plot", 0, 0, 3, gbest, phi_desired, distance);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfEBEformybus(int nargout, mxArray** YIbus, mxArray* linedata, 
                                 mxArray* n)
{
  return mclMlfFeval(_mcr_inst, "EBEformybus", nargout, 1, 2, YIbus, linedata, n);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfEBEinputfile()
{
  return mclMlfFeval(_mcr_inst, "EBEinputfile", 0, 0, 0);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_DED_10(int nargout, mxArray** Total_Value, mxArray** Total_Cost, 
                               mxArray** Total_Penalty, mxArray* Input_Vector, mxArray* 
                               My_Action)
{
  return mclMlfFeval(_mcr_inst, "fn_DED_10", nargout, 3, 2, Total_Value, Total_Cost, Total_Penalty, Input_Vector, My_Action);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_DED_5(int nargout, mxArray** Total_Value, mxArray** Total_Cost, 
                              mxArray** Total_Penalty, mxArray* Input_Vector, mxArray* 
                              Display)
{
  return mclMlfFeval(_mcr_inst, "fn_DED_5", nargout, 3, 2, Total_Value, Total_Cost, Total_Penalty, Input_Vector, Display);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_ELD_13(int nargout, mxArray** Total_Cost, mxArray** Cost, 
                               mxArray** Total_Penalty, mxArray* Input_Population, 
                               mxArray* Display)
{
  return mclMlfFeval(_mcr_inst, "fn_ELD_13", nargout, 3, 2, Total_Cost, Cost, Total_Penalty, Input_Population, Display);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_ELD_140(int nargout, mxArray** Total_Cost, mxArray** Cost, 
                                mxArray** Total_Penalty, mxArray* Input_Population, 
                                mxArray* Display)
{
  return mclMlfFeval(_mcr_inst, "fn_ELD_140", nargout, 3, 2, Total_Cost, Cost, Total_Penalty, Input_Population, Display);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_ELD_15(int nargout, mxArray** Total_Cost, mxArray** Cost, 
                               mxArray** Total_Penalty, mxArray* Input_Population, 
                               mxArray* Display)
{
  return mclMlfFeval(_mcr_inst, "fn_ELD_15", nargout, 3, 2, Total_Cost, Cost, Total_Penalty, Input_Population, Display);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_ELD_40(int nargout, mxArray** Total_Cost, mxArray** Cost, 
                               mxArray** Total_Penalty, mxArray* Input_Population, 
                               mxArray* Display)
{
  return mclMlfFeval(_mcr_inst, "fn_ELD_40", nargout, 3, 2, Total_Cost, Cost, Total_Penalty, Input_Population, Display);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_ELD_6(int nargout, mxArray** Total_Cost, mxArray** Cost, 
                              mxArray** Total_Penalty, mxArray* Input_Population, 
                              mxArray* Display)
{
  return mclMlfFeval(_mcr_inst, "fn_ELD_6", nargout, 3, 2, Total_Cost, Cost, Total_Penalty, Input_Population, Display);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_HT_ELD_Case_1(int nargout, mxArray** Total_Value, mxArray** 
                                      Total_Cost, mxArray** Total_Penalty, mxArray* 
                                      Input_Vector, mxArray* My_Action)
{
  return mclMlfFeval(_mcr_inst, "fn_HT_ELD_Case_1", nargout, 3, 2, Total_Value, Total_Cost, Total_Penalty, Input_Vector, My_Action);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_HT_ELD_Case_2(int nargout, mxArray** Total_Value, mxArray** 
                                      Total_Cost, mxArray** Total_Penalty, mxArray* 
                                      Input_Vector, mxArray* My_Action)
{
  return mclMlfFeval(_mcr_inst, "fn_HT_ELD_Case_2", nargout, 3, 2, Total_Value, Total_Cost, Total_Penalty, Input_Vector, My_Action);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFn_HT_ELD_Case_3(int nargout, mxArray** Total_Value, mxArray** 
                                      Total_Cost, mxArray** Total_Penalty, mxArray* 
                                      Input_Vector, mxArray* My_Action)
{
  return mclMlfFeval(_mcr_inst, "fn_HT_ELD_Case_3", nargout, 3, 2, Total_Value, Total_Cost, Total_Penalty, Input_Vector, My_Action);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFourthh()
{
  return mclMlfFeval(_mcr_inst, "fourthh", 0, 0, 0);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfFourth(int nargout, mxArray** f, mxArray* x, mxArray* u)
{
  return mclMlfFeval(_mcr_inst, "fourth", nargout, 1, 2, f, x, u);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfGetlimit_cassini2(int nargout, mxArray** lower_bound, mxArray** 
                                       upper_bound)
{
  return mclMlfFeval(_mcr_inst, "getlimit_cassini2", nargout, 2, 0, lower_bound, upper_bound);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfGetlimit_messenger(int nargout, mxArray** ub, mxArray** lb)
{
  return mclMlfFeval(_mcr_inst, "getlimit_messenger", nargout, 2, 0, ub, lb);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfHsdrf(int nargout, mxArray** f, mxArray* pop)
{
  return mclMlfFeval(_mcr_inst, "hsdrf", nargout, 1, 1, f, pop);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfIntgrl1(int nargout, mxArray** dy, mxArray* t, mxArray* x, mxArray* 
                             u)
{
  return mclMlfFeval(_mcr_inst, "intgrl1", nargout, 1, 3, dy, t, x, u);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfIntgrl(int nargout, mxArray** dy, mxArray* t, mxArray* x, mxArray* u)
{
  return mclMlfFeval(_mcr_inst, "intgrl", nargout, 1, 3, dy, t, x, u);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMessengerfull(int nargout, mxArray** J, mxArray* t, mxArray* problem)
{
  return mclMlfFeval(_mcr_inst, "messengerfull", nargout, 1, 2, J, t, problem);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMga_dsm(int nargout, mxArray** J, mxArray** DVvec, mxArray** DVrel, 
                             mxArray* t, mxArray* problem)
{
  return mclMlfFeval(_mcr_inst, "mga_dsm", nargout, 3, 2, J, DVvec, DVrel, t, problem);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION11_10(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                      input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION11_10", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION11_5(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                     input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION11_5", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION12_13(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                      input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION12_13", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION12_140(int nargout, mxArray** y, mxArray** Count, 
                                       mxArray* input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION12_140", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION12_15(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                      input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION12_15", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION12_40(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                      input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION12_40", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION12_6(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                     input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION12_6", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION13_1(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                     input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION13_1", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION13_2(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                     input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION13_2", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION13_3(int nargout, mxArray** y, mxArray** Count, mxArray* 
                                     input_array)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION13_3", nargout, 2, 1, y, Count, input_array);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION14(int nargout, mxArray** F, mxArray* x)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION14", nargout, 1, 1, F, x);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfMY_FUNCTION15(int nargout, mxArray** F, mxArray* x)
{
  return mclMlfFeval(_mcr_inst, "MY_FUNCTION15", nargout, 1, 1, F, x);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfTersoff(int nargout, mxArray** f, mxArray* x)
{
  return mclMlfFeval(_mcr_inst, "tersoff", nargout, 1, 1, f, x);
}

LIB_CEC2011_C_API 
bool MW_CALL_CONV mlfTrapezoidalcir(int nargout, mxArray** q, mxArray* x2, mxArray* 
                                    upper, mxArray* lower, mxArray* N1, mxArray* 
                                    phi_desired, mxArray* distance, mxArray* dim)
{
  return mclMlfFeval(_mcr_inst, "trapezoidalcir", nargout, 1, 7, q, x2, upper, lower, N1, phi_desired, distance, dim);
}
