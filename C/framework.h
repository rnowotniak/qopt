#ifndef _FRAMEWORK_H
#define _FRAMEWORK_H 1

#include <cstdlib>
#include <exception>
#include <string.h>


// XXX
#define QOPT_PATH "/home/rob/qopt/"


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
		virtual void (repairer) (ARGTYPE*, int) { }

};

inline bool matches(const char *chromo, const char *schema, int length) {
	for (int i = 0; i < length; i++) {
		if (schema[i] != '*' && schema[i] != chromo[i]) {
			return false;
		}
	}
	return true;
}


inline void dec2bin(char *buf, long int dec, int length) {
	memset(buf, '0', length);
	int i = length - 1;
	while (dec > 0) {
		buf[i] = '0' + dec % 2;
		dec = dec >> 1;
		i--;
	}
}

inline void dec2ternary(char *buf, long int dec, int length) {
	memset(buf, '0', length);
	int i = length - 1;
	while (dec > 0) {
		buf[i] = '0' + dec % 3;
		if (buf[i] == '2') {
			buf[i] = '*';
		}
		dec = dec / 3;
		i--;
	}
}

inline void dec2four(char *buf, long int dec, int length) {
	memset(buf, '0', length);
	int i = length - 1;
	while (dec > 0) {
		buf[i] = '0' + dec % 4;
		//if (buf[i] == '2') {
			//buf[i] = '*';
		//}
		dec = dec / 4;
		i--;
	}
}

inline int order(char *s, int len) {
	int result = 0;
	for (int i = 0; i < len; i++) {
		if (s[i] != '*') {
			result++;
		}
	}
	return result;
}

inline int deflenth(char *s, int len) {
	int start = -1, stop;
	for (int i = 0; i < len; i++) {
		if (s[i] != '*') {
			start = i;
			break;
		}
	}
	if (start == -1) {
		return -1;
	}
	for (int i = len - 1; i >= 0; i--) {
		if (s[i] != '*') {
			stop = i;
			break;
		}
	}
	return stop - start;
}

inline unsigned sample(const char *schema, int length) {
	char s[length + 1];
    s[length] = '\0';
	for (int i = 0; i < length; i++) {
		if (schema[i] != '*') {
			s[i] = schema[i];
			continue;
		}
		s[i] = ((1.f * rand() / RAND_MAX) > .5) ? '1' : '0';
	}
	return strtoul(s, NULL, 2);
}

#endif

