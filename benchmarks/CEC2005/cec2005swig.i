/*
 * Moze to sie da zintegrowac z cec2005.py ?
 * Moze ponizsze getfun daloby sie wrzucic do funwrapper i wystawic getfun przez ctypes?
 *
 */

%module cec2005swig

%{
#define SWIG_FILE_WITH_INIT

#include <dlfcn.h>
#include <stdio.h>

typedef long double (*evaluator_ptr)(long double *, int);

evaluator_ptr getfun(int which) {
        // XXX how to get the proper directory here?
        // XXX - The problem no loger exists probably, as there is chdir in 'evaluate' initialization
        char buf[FILENAME_MAX] = { '\0' };
        snprintf(buf, FILENAME_MAX - 1, "%s/libf%d.so", ABSDIR, which);  // XXX
        void *handle = dlopen(buf, 0);
        void *f = dlsym(handle, "evaluate");
        printf("%p\n", f);
        // XXX dokonczyc
        return (evaluator_ptr) f;
}
%}

typedef long double (*evaluator_ptr)(long double *, int);

extern evaluator_ptr getfun(int which);

