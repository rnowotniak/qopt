#ifndef _mCEC_FUNCTION_H
#define mCEC_FUNCTION_H
typedef double FIELD_TYPE;
int Initial_CEC2011_Cost_Function(void);
const int _HAS_ERROR = 1;
const int _NO_ERROR = 0;

void Terminate_CEC2011_Cost_Function(void);

void cost_function1(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function2(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function3(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function4(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function5(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function6(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function7(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function8(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function9(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function10(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function11_5(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function11_10(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function12_6(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function12_13(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function12_15(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function12_40(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function12_140(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function13_1(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function13_2(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function13_3(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function14(FIELD_TYPE *x, FIELD_TYPE *f);
void cost_function15(FIELD_TYPE *x, FIELD_TYPE *f);

#endif
