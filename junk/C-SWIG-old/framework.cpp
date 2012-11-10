#include "framework.h"

template <class DTYPE>
EA<DTYPE>::EA(int popsize = 0, int chromlen = 0) {
	printf("EA constructor\n");

	// some default values for EA
	this->popsize = popsize;
	this->chromlen = chromlen;
	this->evaluator = 0;
	this->evaluation_counter = 0;
	this->t = 0;
	this->tmax = 0; // XXX
	this->best = (Individual<DTYPE>*)0;
	// XXX move it to some proper place
	srand(time(0));
    
    if (popsize > 0 && chromlen > 0) {
        printf("Allocating P.\n");
        this->P = new DTYPE *[this->popsize];
        for (int i = 0; i < this->popsize; i++) {
            P[i] = new DTYPE[this->chromlen];
        }
    }
}

template <class DTYPE>
EA<DTYPE>::EA(const EA<DTYPE> &orig) {
	printf("EA copy constructor\n");
	this->chromlen = orig.chromlen;
	this->popsize = orig.popsize;
	this->evaluator = orig.evaluator;
	this->evaluation_counter = orig.evaluation_counter;
	this->t = orig.t;
	this->tmax = orig.tmax;

	this->P = (DTYPE**)0;
	if (orig.P != 0) {
		this->P = new DTYPE *[this->popsize];
		for (int i = 0; i < this->popsize; i++) {
			this->P[i] = new DTYPE[this->chromlen];
			memcpy(this->P[i], orig.P[i], sizeof(DTYPE) * chromlen);
		}
	}

	this->best = (Individual<DTYPE>*)0;
	if (orig.best != 0) {
		this->best = new Individual<DTYPE>(*orig.best);
	}
}

template <class DTYPE>
EA<DTYPE>::~EA() {
	printf("EA destructor\n");
	if (this->best) {
		delete this->best;
	}
	if (this->P) {
		for (int i = 0; i < this->popsize; i++) {
			delete [] P[i];
		}
		delete [] this->P;
	}
}

template <class DTYPE>
void EA<DTYPE>::saveGeneration() {
	EA<DTYPE> *copy = this->clone();
	copy->history.clear();
	history.push_back(copy);
}

template <class DTYPE>
void EA<DTYPE>::initialize() {
	printf("Initializing EA.\n");

	// XXX test
	for (int i = 0; i < this->popsize; i++) {
		for (int j = 0; j < this->chromlen; j++) {
			P[i][j] = i + j;
		}
	}
}

template <class DTYPE>
void EA<DTYPE>::generation() {
	assert(this->P != 0);
	printf("Generation %d\n", this->t);
	// evaluation
	DTYPE *fvalues = this->evaluate(this->P, this->popsize, this->chromlen);
	// store the best
	if (best) {
		delete best;
	}
	best = new Individual<DTYPE>(fvalues[0], P[0], this->chromlen); // XXX find actual best
	// operators
	this->operators();
}

template <class DTYPE>
void EA<DTYPE>::operators() {
	// do nothing
}

template <class DTYPE>
DTYPE* EA<DTYPE>::evaluate(DTYPE **population, int popsize, int chromlen) {
	DTYPE *results = new DTYPE[popsize]; // dealloc XXX  (use some fvalues buffer)
	for (int i = 0; i < popsize; i++) {
		results[i] = this->evaluator(population[i], chromlen);
	}
	return results;
}

template <class DTYPE>
void EA<DTYPE>::run() {
	printf("Running...\n");
	this->t = 0;
	this->initialize();
	this->saveGeneration();
	while (!this->termination()) {
		this->t += 1;
		this->generation();
		this->saveGeneration();
	}
}

template <class DTYPE>
bool EA<DTYPE>::termination() {
	return this->t >= tmax;
}

template class EA<long double>;

