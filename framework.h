#ifndef _FRAMEWORK_H
#define _FRAMEWORK_H 1

#include <exception>

class QOptException : public std::exception {

	const char *str;

	public:

	QOptException(const char *str) : str(str) { }

	virtual const char *what() const throw()
	{
		return str;
	}
};

template <class ARGTYPE, class RESTYPE>
class Problem {

	public:

		virtual RESTYPE evaluator (ARGTYPE *, int) = 0;
		virtual void (repairer) (ARGTYPE*, int) {
			// throw QOptException("Repairer has not been implemented");
		}
};

#endif

