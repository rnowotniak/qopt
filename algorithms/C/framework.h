#ifndef _FRAMEWORK_H
#define _FRAMEWORK_H

#include <cstdio>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <ctime>

template <class DTYPE>
class Individual {
	public:
		DTYPE fitness;
		DTYPE *genotype;
		int chromlen;

		Individual(double fitness) {
			this->fitness = (DTYPE) fitness;
			this->genotype = NULL;
			this->chromlen = 0;
		}

		Individual(DTYPE fitness, DTYPE *genotype, int chromlen) {
			this->fitness = fitness;
			this->chromlen = chromlen;
			this->genotype = 0;
			if (genotype != 0 && chromlen > 0) {
				this->genotype = new DTYPE[chromlen];
				memcpy(this->genotype, genotype, sizeof(DTYPE) * chromlen);
			}
		}

		Individual(const Individual<DTYPE> & orig) {
			Individual(orig.fitness, orig.genotype, orig.chromlen);
		}

		~Individual() {
			if (this->genotype) {
				delete [] this->genotype;
			}
		}
};


template <class DTYPE>
class EA {
	private:

		std::vector<EA<DTYPE>* > history;

		void saveGeneration();

	protected:

		virtual EA<DTYPE>* clone() const = 0;

	public:

		/* P nie musi byc uzywane, jesli klasa dziedziczaca nie wywola EA::initialize()
		 * i gdy nadpisze generation() (nie uzywajac P).
		 * Jesli jest uzywane, to musi byc tablica popsize x chromlen */
		DTYPE **P;

		int popsize;

		int chromlen;

		DTYPE (*evaluator) (DTYPE *, int);

		int evaluation_counter;

		int tmax;

		Individual<DTYPE> *best; // best ever, updated in EA::generation()

		int t;

		// Constructors, destructor

		EA(int popsize, int chromlen); // allocates memmory for P  (popsize x chromlen)

		~EA();

		EA(const EA<DTYPE> &orig);

		// METHODS

		virtual void initialize() = 0;

		void run();

		virtual void generation();

		virtual void operators();

		virtual DTYPE* evaluate(DTYPE **population, int popsize, int chromlen);

		virtual bool termination();

		EA<DTYPE> *getGeneration(int n) {
			return history[n];
		}
};

#endif

