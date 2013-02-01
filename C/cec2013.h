#ifndef _CEC2013_H
#define _CEC2013_H 1

#include "framework.h"

#include <unistd.h>
#include "test_func.h"

class CEC2013 : public Problem<double,double> {

	public:

		const int fnum;

		CEC2013(int fnum) : fnum(fnum) { }

		virtual double evaluator(double *x, int length) {
			double result;
			char oldpath[256] = {'\0'};
			getcwd(oldpath, sizeof(oldpath) - 1);
			chdir(QOPT_PATH "/problems/CEC2013/cec13ccode");
			test_func(x, &result, length, 1, fnum);
			chdir(oldpath);
			return result;
		}
};


#endif
