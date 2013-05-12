
#include <vector>
#include <list>
#include <utility>
#include <algorithm>
#include "qiea2.h"

using namespace std;

void QIEA2::run() {
	t = 0;
	int tmax = 100000 / K;
	initialize();
	observe(); //
	evaluate(); //
	storebest(); //
	while (t < tmax) {
		// printf("Generation %d\n", t);
		// printf("bestval: %f\n", bestval);
		t++;
		observe();
		//evaluate();
		update();
		storebest();
	}
}

#define Qij (Q + i * (5 * (chromlen/2)) + (5 * j))
#define Pij (P + i * (chromlen) + (2 * j))

void QIEA2::initialize() {
	bestval = std::numeric_limits<DTYPE>::max();
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {
			Qij[0] = (1. * rand() / RAND_MAX) * 200. - 100.;   // X XXX
			Qij[1] = (1. * rand() / RAND_MAX) * 200. - 100.;   // Y XXX
			Qij[2] = M_PI * rand() / RAND_MAX; // orientation
			Qij[3] = 40. * rand() / RAND_MAX; // scale X   XXX 40 param
			Qij[4] = 40. * rand() / RAND_MAX; // scale Y   XXX 40 param
		}
	}
}

void QIEA2::observe() {
	// Firstly, sample each distribution in Q one by one
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {
			double u = Qij[3] * box_muller();
			double v = Qij[4] * box_muller();
			double theta = Qij[2];

			double u2 = u * cos(theta) - v * sin(theta);
			double v2 = u * sin(theta) + v * cos(theta);
			u = u2;
			v = v2;
			u += Qij[0];
			v += Qij[1];
			Pij[0] = u;
			Pij[1] = v;
		}
	}
	// if K > |Q|, sample random distributions in Q until K is reached
	for (int k = popsize; k < K; k++) {
		// choose a random element from Q
		int i = (int)((1. * rand() / RAND_MAX) * popsize);

		// observe it
		for (int j = 0; j < chromlen / 2; j++) {
			double u = Qij[3] * box_muller();
			double v = Qij[4] * box_muller();
			double theta = Qij[2];
			double u2 = u * cos(theta) - v * sin(theta);
			double v2 = u * sin(theta) + v * cos(theta);
			u = u2;
			v = v2;
			u += Qij[0];
			v += Qij[1];

			(P + k * (chromlen) + (2 * j))[0] = u;
			(P + k * (chromlen) + (2 * j))[1] = v;
		}
	}
}

void QIEA2::storebest() {
	DTYPE val = std::numeric_limits<DTYPE>::max();
	int i_best;
	for (int i = 0; i < K; i++) {
		if (fvals[i] < val) { // XXX minmax
			val = fvals[i];
			i_best = i;
		}
	}
	if (val < bestval) { /// XXX minmax
		bestval = val;
		memcpy(best, P + i_best * chromlen, sizeof(DTYPE) * chromlen);
	}
}

// evaluate P and store its score in fvals
void QIEA2::evaluate() {
	for (int i = 0; i < K; i++) {
		fvals[i] = problem->evaluator(P + i * chromlen, chromlen);
	}
}

// crossover mieszajacy miedzy P i P_old
//   i zapis wyniku w P
void QIEA2::crossover() {

	DTYPE *p_ptr = P;
	DTYPE *o_ptr = P_old;

	for (int i = 0; i < K; i++) {
		for (int j = 0; j < chromlen; j++) {
			float r = 1.0 * rand() / RAND_MAX;
			if (r < XI) {
				*p_ptr = *o_ptr;
			}

			p_ptr++;
			o_ptr++;
		}
	}

}

// crossover P   --  1-point crossover of P population
/*
void QIEA2::crossover() {

	vector<int> toCrossover;

	for (int i = 0; i < K; i++) {
		float r = 1.0 * rand() / RAND_MAX;
		if (r < XI) {
			toCrossover.push_back(i);
		}
	}

	if (toCrossover.size() % 2 != 0) {
		vector<int>::iterator it1 = toCrossover.begin();
		it1 += (int)(1.0 * rand() / RAND_MAX * (toCrossover.size()));
		// advance(it1, ...);
		toCrossover.erase(it1);
	}

	// make pairs
	vector<pair<int,int> > pairs;
	while (toCrossover.size() > 0) {
		vector<int>::iterator it1;
			
		it1 = toCrossover.begin();
		it1 += (int)(1.0 * rand() / RAND_MAX * (toCrossover.size()));
		int ind1 = *it1;
		toCrossover.erase(it1);

		it1 = toCrossover.begin();
		it1 += (int)(1.0 * rand() / RAND_MAX * (toCrossover.size()));
		int ind2 = *it1;
		toCrossover.erase(it1);

		pairs.push_back(pair<int,int>(ind1, ind2));
	}

	for (int pind = 0; pind < pairs.size(); pind++) {
		pair<int,int> p = pairs[pind];
		int i = p.first;
		int j = p.second;
		int xpoint = (int)(1.0 * rand() / RAND_MAX * chromlen);
		DTYPE buf[chromlen];
		memcpy(buf, P + i * chromlen, sizeof(DTYPE) * chromlen);
		memcpy(P + i * chromlen, P + j * chromlen, sizeof(DTYPE) * xpoint);
		memcpy(P + j * chromlen, buf, sizeof(DTYPE) * xpoint);
	}
}
*/

void QIEA2::mutate() {
	/// XXX implement
}

bool cmp(pair<int,int> f, pair<int,int> s) {
	return f.second < s.second;
}

void QIEA2::update() {

	// P <- K best from {P_old + P}
	DTYPE P_both[K * chromlen * 2];
	DTYPE fvals_both[K * 2];
	vector<pair<int,int> > pairs;

	if (t == 1) {
		// P_old is not set yet
		memcpy(P_both, P, sizeof(DTYPE) * chromlen * K);
		evaluate(); // put new P scores into fval table
		for (int n = 0; n < K; n++) {
			pairs.push_back(pair<int,int>(n,fvals[n]));
		}
	}
	else {
		crossover();
		mutate();

		//    1) P_both  <-  P + P_old
		//       fvals_both <- fvals + fvals_old
		memcpy(P_both, P_old, sizeof(DTYPE) * chromlen * K);
		memcpy(P_both + K * chromlen, P, sizeof(DTYPE) * chromlen * K);
		memcpy(fvals_both, fvals, sizeof(DTYPE) * K);
		evaluate(); // put new P scores into fval table
		memcpy(fvals_both + K, fvals, sizeof(DTYPE) * K);

		for (int n = 0; n < 2*K; n++) {
			pairs.push_back(pair<int,int>(n,fvals_both[n]));
		}
	}

		//    2) sort P_both
	sort(pairs.begin(), pairs.end(), cmp);
	// result: pairs contains sorted P_both

	// Now, update Q
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {

			// update Q according to |Q| best in P
			//Qij[0] = (best + 2*j)[0];  //  converge to the best ever
			//Qij[1] = (best + 2*j)[1];  //  converge to the best ever
			Qij[0] = (P_both + pairs[i].first * chromlen)[2*j];
			Qij[1] = (P_both + pairs[i].first * chromlen)[2*j + 1];

			if (i == 0 && j == 0) {
				//printf("-> %g\n", Qij[3]);
			}

			//Qij[2] += .1; // rotating
			//Qij[3] *= delta;
			//Qij[4] *= delta;
		}
	}

	// update sampling width, count how many individuals have improved
	if (t > 1) {
		vector<DTYPE> list1, list2;
		int improved = 0;
		for (int i = 0; i < K; i++) {
			list1.push_back( fvals_both[i] ); // old
			list2.push_back( fvals_both[K + i] ); // new
		}
		sort(list1.begin(), list1.end());
		sort(list2.begin(), list2.end());
		for (int i = 0; i < K; i++) {
			if (list2[i] < list1[i]) { // XXX minmax
				improved++;
			}
		}
		float FI = .2;
		for (int i = 0; i < popsize; i++) {
			for (int j = 0; j < chromlen / 2; j++) {
				if (1.0 * improved / K > FI) {
					if (Qij[3] > 0.00001)
						Qij[3] *= delta;
					if (Qij[4] > 0.00001)
						Qij[4] *= delta;
				}
				else if (1.0 * improved / K < FI) {
					if (Qij[3] < 40)
						Qij[3] /= delta;
					if (Qij[4] < 40)
						Qij[4] /= delta;
				}
			}
		}
	}

	// P_old <- P
	memcpy(P_old, P, sizeof(DTYPE) * K * chromlen);
}


/*
#include "cec2013.h"
#include <time.h>
int main() {
	srand(time(0));
	//srand(5);//time(0));
	int dim = 10;
	int popsize = 10;
	QIEA2 *qiea2 = new QIEA2(dim, popsize);
	//for (int i = 0; i < dim; i++) {
	//	qiea2->bounds[i][0] = -100;
	//	qiea2->bounds[i][1] = 100;
	//}

	qiea2->XI = 0;
	qiea2->delta = .999;
	Problem<double,double> *fun = new CEC2013(1);
	// double x[2] = {-39.3, 58.8};
	// double val = fun->evaluator(x, 2);
	// printf("-> %f\n", val);
	// return 0;
	qiea2->problem = fun;
	qiea2->run();
	printf("Final bestval: %f\n", qiea2->bestval);
	printf("Final best: ");
	for (int i = 0; i < dim; i++) {
		printf("%f, ", qiea2->best[i]);
	}
	printf("\n");
}
*/

