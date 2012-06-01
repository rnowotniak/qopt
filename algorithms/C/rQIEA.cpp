#include "rQIEA.h"

template <class DTYPE>
rQIEA<DTYPE>::rQIEA(int popsize, int chromlen) : EA<DTYPE>::EA(popsize, chromlen) {
	printf("rQIEA constructor\n");
	Pc = 0.6;

	Q = new float **[popsize];
	for (int i = 0; i < popsize; i++) {
		Q[i] = new float*[chromlen];
		for (int j = 0; j < chromlen; j++) {
			Q[i][j] = new float[2];
		}
	}

	bounds = new DTYPE* [chromlen];
    for (int i = 0; i < chromlen; i++) {
        bounds[i] = new DTYPE[2];
        bounds[i][0] = -130;
        bounds[i][1] = 130;
    }

    qbest = new float* [chromlen];

}

template <class DTYPE>
rQIEA<DTYPE>::rQIEA(const rQIEA<DTYPE> & orig) : EA<DTYPE>::EA(orig) {
	printf("rQIEA copy constructor\n");
	this->Pc = orig.Pc;

	Q = new float **[this->popsize];
	for (int i = 0; i < this->popsize; i++) {
		Q[i] = new float*[this->chromlen];
		for (int j = 0; j < this->chromlen; j++) {
			Q[i][j] = new float[2];

			Q[i][j][0] = orig.Q[i][j][0];
			Q[i][j][1] = orig.Q[i][j][1];
		}
	}
}

template <class DTYPE>
rQIEA<DTYPE>::~rQIEA() {
	printf("rQIEA destructor\n");

	for (int i = 0; i < this->popsize; i++) {
		for (int j = 0; j < this->chromlen; j++) {
            delete [] this->Q[i][j];
        }
        delete [] this->Q[i];
    }

    delete [] this->Q;

    for (int i = 0; i < this->chromlen; i++) {
        delete [] bounds[i];
    }
    delete [] bounds;
}

template <class DTYPE>
EA<DTYPE>* rQIEA<DTYPE>::clone() const {
	return new rQIEA<DTYPE>(*this);
}

template <class DTYPE>
void rQIEA<DTYPE>::initialize() {
	printf("Initializing rQIEA.\n");

	for (int i = 0; i < this->popsize; i++) {
		for (int j = 0; j < this->chromlen; j++) {
			float alpha = 2.f * (1.f * rand() / RAND_MAX) - 1;
			float beta = sqrtf(1 - alpha * alpha) * ((rand() % 2) ? -1 : 1);
			Q[i][j][0] = alpha;
			Q[i][j][1] = beta; // i + j (test)
		}
	}
}

template <class DTYPE>
void rQIEA<DTYPE>::operators() {
	// XXX do nothing?
}

template <class DTYPE>
void rQIEA<DTYPE>::observe(float ***Q, int popsize, int chromlen) {
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen; j++) {
			float r = 1.f * rand() / RAND_MAX;
			if (r <= .5) {
				this->P[i][j] = Q[i][j][0] * Q[i][j][0];
			} else {
				this->P[i][j] = Q[i][j][1] * Q[i][j][1];
			}
			this->P[i][j] *= (bounds[j][1] - bounds[j][0]);
			this->P[i][j] += bounds[j][0];
		}
	}
}

template <class DTYPE>
void rQIEA<DTYPE>::generation() {
	printf("Generation: %d\n", this->t);

	observe(this->Q, this->popsize, this->chromlen);

	DTYPE *fvalues = evaluate(this->P, this->popsize, this->chromlen);

	if (this->best) {
		delete this->best;
	}
	this->best = new Individual<DTYPE>(fvalues[0]);

	updateQ();

	recombination();

	return;
	long double aa[] = { 2, 3 };
	printf("gen: %d, fvalue: %Lf\n", this->t, this->evaluator(aa, 2));
}

// XXX implement it, and move it somewhere else?
float LUT(float alpha, float beta, float alphabest, float betabest) {
	// XXX implement this
	return 1;
}

template <class DTYPE>
void rQIEA<DTYPE>::updateQ() {
	for (int i = 0; i < this->popsize; i++) {
		for (int j = 0; j < this->chromlen; j++) {
			float *q = Q[i][j];
			float qprim[2];
			float k = M_PI / (100 + this->t % 100);
			float theta = k * LUT(q[0], q[1], qbest[j][0], qbest[j][1]);

			qprim[0] = q[0] * cos(theta) + q[1] * (-sin(theta));
			qprim[1] = q[0] * sin(theta) + q[1] * (cos(theta));
			q[0] = qprim[0];
			q[1] = qprim[1];
		}
	}
}

template <class DTYPE>
void rQIEA<DTYPE>::recombination() {
	// XXX implement this
}

// template instantiation
template class rQIEA<long double>;

