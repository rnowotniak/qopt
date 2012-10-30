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

#ifndef __CEC2011_h
#define __CEC2011_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SUNPRO_CC)
/* Solaris shared libraries use __global, rather than mapfiles
 * to define the API exported from a shared library. __global is
 * only necessary when building the library -- files including
 * this header file to use the library do not need the __global
 * declaration; hence the EXPORTING_<library> logic.
 */

#ifdef EXPORTING_CEC2011
#define PUBLIC_CEC2011_C_API __global
#else
#define PUBLIC_CEC2011_C_API /* No import statement needed. */
#endif

#define LIB_CEC2011_C_API PUBLIC_CEC2011_C_API

#elif defined(_HPUX_SOURCE)

#ifdef EXPORTING_CEC2011
#define PUBLIC_CEC2011_C_API __declspec(dllexport)
#else
#define PUBLIC_CEC2011_C_API __declspec(dllimport)
#endif

#define LIB_CEC2011_C_API PUBLIC_CEC2011_C_API


#else

#define LIB_CEC2011_C_API

#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_CEC2011_C_API 
#define LIB_CEC2011_C_API /* No special import/export declaration */
#endif

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV CEC2011InitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV CEC2011Initialize(void);

extern LIB_CEC2011_C_API 
void MW_CALL_CONV CEC2011Terminate(void);



extern LIB_CEC2011_C_API 
void MW_CALL_CONV CEC2011PrintStackTrace(void);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxAngle3d(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxAntennafunccircular(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                         *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxArray_factorcir(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxBench_func(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxBounds(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxCassini2(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxCost_fn(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxData6Bus(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxDiffsolv(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxDisplay_plot(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxEBEformybus(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxEBEinputfile(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_DED_10(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_DED_5(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_13(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_140(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_15(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_40(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_ELD_6(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_HT_ELD_Case_1(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_HT_ELD_Case_2(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFn_HT_ELD_Case_3(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFourthh(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxFourth(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxGetlimit_cassini2(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                       *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxGetlimit_messenger(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                        *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxHsdrf(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxIntgrl1(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxIntgrl(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMessengerfull(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMga_dsm(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION11_10(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION11_5(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_13(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_140(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                       *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_15(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_40(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                      *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION12_6(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION13_1(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION13_2(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION13_3(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION14(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxMY_FUNCTION15(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxTersoff(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
bool MW_CALL_CONV mlxTrapezoidalcir(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_CEC2011_C_API 
long MW_CALL_CONV CEC2011GetMcrID();



extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfAngle3d(int nargout, mxArray** th, mxArray* x, mxArray* j, mxArray* i, mxArray* k);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfAntennafunccircular(int nargout, mxArray** y, mxArray** sllreturn, mxArray** bwfn, mxArray* x1, mxArray* null, mxArray* phi_desired, mxArray* distance);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfArray_factorcir(int nargout, mxArray** y, mxArray* x1, mxArray* phi, mxArray* phi_desired, mxArray* distance, mxArray* dim);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfBench_func(int nargout, mxArray** f, mxArray* x, mxArray* fun_num);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfBounds();

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfCassini2(int nargout, mxArray** J, mxArray* t, mxArray* problem);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfCost_fn(int nargout, mxArray** fitness, mxArray** PENALTY, mxArray** rate_d, mxArray* x);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfData6Bus();

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfDiffsolv(int nargout, mxArray** dy, mxArray* t, mxArray* x, mxArray* u);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfDisplay_plot(mxArray* gbest, mxArray* phi_desired, mxArray* distance);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfEBEformybus(int nargout, mxArray** YIbus, mxArray* linedata, mxArray* n);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfEBEinputfile();

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_DED_10(int nargout, mxArray** Total_Value, mxArray** Total_Cost, mxArray** Total_Penalty, mxArray* Input_Vector, mxArray* My_Action);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_DED_5(int nargout, mxArray** Total_Value, mxArray** Total_Cost, mxArray** Total_Penalty, mxArray* Input_Vector, mxArray* Display);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_ELD_13(int nargout, mxArray** Total_Cost, mxArray** Cost, mxArray** Total_Penalty, mxArray* Input_Population, mxArray* Display);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_ELD_140(int nargout, mxArray** Total_Cost, mxArray** Cost, mxArray** Total_Penalty, mxArray* Input_Population, mxArray* Display);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_ELD_15(int nargout, mxArray** Total_Cost, mxArray** Cost, mxArray** Total_Penalty, mxArray* Input_Population, mxArray* Display);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_ELD_40(int nargout, mxArray** Total_Cost, mxArray** Cost, mxArray** Total_Penalty, mxArray* Input_Population, mxArray* Display);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_ELD_6(int nargout, mxArray** Total_Cost, mxArray** Cost, mxArray** Total_Penalty, mxArray* Input_Population, mxArray* Display);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_HT_ELD_Case_1(int nargout, mxArray** Total_Value, mxArray** Total_Cost, mxArray** Total_Penalty, mxArray* Input_Vector, mxArray* My_Action);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_HT_ELD_Case_2(int nargout, mxArray** Total_Value, mxArray** Total_Cost, mxArray** Total_Penalty, mxArray* Input_Vector, mxArray* My_Action);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFn_HT_ELD_Case_3(int nargout, mxArray** Total_Value, mxArray** Total_Cost, mxArray** Total_Penalty, mxArray* Input_Vector, mxArray* My_Action);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFourthh();

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfFourth(int nargout, mxArray** f, mxArray* x, mxArray* u);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfGetlimit_cassini2(int nargout, mxArray** lower_bound, mxArray** upper_bound);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfGetlimit_messenger(int nargout, mxArray** ub, mxArray** lb);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfHsdrf(int nargout, mxArray** f, mxArray* pop);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfIntgrl1(int nargout, mxArray** dy, mxArray* t, mxArray* x, mxArray* u);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfIntgrl(int nargout, mxArray** dy, mxArray* t, mxArray* x, mxArray* u);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMessengerfull(int nargout, mxArray** J, mxArray* t, mxArray* problem);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMga_dsm(int nargout, mxArray** J, mxArray** DVvec, mxArray** DVrel, mxArray* t, mxArray* problem);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION11_10(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION11_5(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION12_13(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION12_140(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION12_15(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION12_40(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION12_6(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION13_1(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION13_2(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION13_3(int nargout, mxArray** y, mxArray** Count, mxArray* input_array);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION14(int nargout, mxArray** F, mxArray* x);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfMY_FUNCTION15(int nargout, mxArray** F, mxArray* x);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfTersoff(int nargout, mxArray** f, mxArray* x);

extern LIB_CEC2011_C_API bool MW_CALL_CONV mlfTrapezoidalcir(int nargout, mxArray** q, mxArray* x2, mxArray* upper, mxArray* lower, mxArray* N1, mxArray* phi_desired, mxArray* distance, mxArray* dim);

#ifdef __cplusplus
}
#endif
#endif
