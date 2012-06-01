#include <cstdio>
#include <cstdlib>

extern "C" {
	double somevar = 15;
	double tmax = 5;
}

class Algorytm {
	public:
		Algorytm() {
			printf("konsruktor Alg\n");
		}
		void initialize();
		void generation();
		void run();

};


void Algorytm::run() {
	printf("run\n");
	this->initialize();
	int t = 0;
	while (t < tmax) {
		t++;
		printf("generation %d\n", t);
		this->generation();
	}
}

void Algorytm::generation() {
	printf("(operators)\n");
}

void Algorytm::initialize() {
	printf("initialize\n");
	printf("somvar value: %g\n", somevar);
}

extern "C" {
	Algorytm *self = NULL;

	void init() {
		self = new Algorytm();
	}

	void run() {
		self->run();
	}

	int aaa() {
		return 13;
	}

    double (*evaluator)(double *, int n);

	void foo(double(*cb)(double *, int)) {
		double arr[2] = {5,6};
		double r = evaluator(arr, 2);
		printf("result: %g\n", r);
	}
}

