#ifndef _CEC2005_H
#define _CEC2005_H 1

#include "framework.h"

#include <cstdio>
#include <unistd.h>
#include <dlfcn.h>

class CEC2005 : public Problem<double,double> {

	public:

		const int fnum;

		long double (*fun)(long double *, int);
		void *handle;

		CEC2005(int fnum) : fnum(fnum) {
			char libfname[256];
			sprintf(libfname, "../problems/CEC2005/libf%d.so", fnum);
			handle = dlopen(libfname, RTLD_NOW);
			if (!handle) {
				printf("err: %s\n", dlerror());
			}
			assert(handle);
			*(void **) (&fun) = dlsym(handle, "evaluate");
			assert(fun);
		}

		~CEC2005() {
			dlclose(handle);
		}

		virtual double evaluator(double *x, int length) {
			double result;
			long double arg[length];
			for (int i = 0; i < length; i++) {
				arg[i] = x[i];
			}
			result = fun(arg, length);
				
			return result;
		}
};


#endif
