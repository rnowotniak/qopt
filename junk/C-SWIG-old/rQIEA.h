#ifndef _RQIEA_H
#define _RQIEA_H
#include "framework.h"

template <class DTYPE>
class rQIEA : public EA<DTYPE> {

	protected:
		void updateQ();

		void recombination();

		virtual void observe(float ***Q, int popsize, int chromlen);

	public:
		virtual EA<DTYPE>* clone() const;
        
		float ***Q;

		float Pc;

        float **qbest;

		DTYPE **bounds;

		rQIEA(int popsize, int chromlen);

		rQIEA(const rQIEA<DTYPE> & orig);

		~rQIEA();

		virtual void initialize();

		virtual void operators();

		virtual void generation();

};


#endif
