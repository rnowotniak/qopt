#ifndef _FRAMEWORK_H
#define _FRAMEWORK_H 1

class Problem {

	public:

		virtual float (evaluator) (char*, int) = 0;
		virtual void (repairer) (char*, int) = 0;
		virtual long double (r_evaluator)(long double *, int) = 0;

};

#endif

