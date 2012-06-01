%module algorithms

%{
#define SWIG_FILE_WITH_INIT
#include "framework.h"
#include "rQIEA.h"

#include <numpy/arrayobject.h>
%}

%init %{
        import_array();
%}


/*
 * Individual
 */
template <class DTYPE>
class Individual {
	public:
		DTYPE fitness;
		DTYPE *genotype;
		int chromlen;

		Individual(double fitness);
		Individual(DTYPE fitness, DTYPE *genotype, int chromlen);
		Individual(const Individual<DTYPE> & orig);
		~Individual();
};

%typemap(out) long double fitness {
        int dimensions[1];
        dimensions[0] = 1;
        PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(1, dimensions, NPY_LONGDOUBLE);
        ((long double*) ar->data)[0] = arg1->fitness;
        $result = PyArray_Scalar(PyArray_DATA(ar), PyArray_DESCR(ar), (PyObject*) ar);
}
%template(Individual_ld) Individual<long double>;
%clear long double;

/*
 * Algorithms
 */
template <class DTYPE>
class EA {
	public:
		EA();
		//DTYPE **P;
		const int popsize;
		const int chromlen;
		DTYPE (*evaluator) (DTYPE *, int);
		int evaluation_counter;
		int tmax;
		Individual<DTYPE> *best;
		int t;

		void run();
		virtual void initialize() = 0;
		virtual void generation();
		virtual void operators();
		virtual bool termination();
};

template <class DTYPE>
class rQIEA : public EA<DTYPE> {
	protected:
		void updateQ();
		void recombination();
	public:

		virtual EA<DTYPE>* clone() const;
		float Pc;
                // double **bounds;
		rQIEA(int popsize, int chromlen);
		~rQIEA();
		virtual void initialize();
		virtual void operators();
		rQIEA<DTYPE> *getGeneration(int n);
};

/*
%typemap(out) float ***Q {
        int dimensions[3];
        dimensions[0] = arg1->popsize;
        dimensions[1] = arg1->chromlen;
        dimensions[2] = 2;
        PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(3, dimensions, NPY_LONGDOUBLE);
        $result = (PyObject*) ar;
}
%typemap(out) float **P {
        int dimensions[2];
        dimensions[0] = arg1->popsize;
        dimensions[1] = arg1->chromlen;
        PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(2, dimensions, NPY_LONGDOUBLE);
        $result = (PyObject*) ar;
}
*/

%template(EA_ld) EA<long double>;
%template(rQIEA_ld) rQIEA<long double>;




/*
%typemap(out) float ***Q {
        int dimensions[3];
        dimensions[0] = arg1->popsize;
        dimensions[1] = arg1->chromlen;
        dimensions[2] = 2;
        PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(3, dimensions, NPY_LONGDOUBLE);
        $result = (PyObject*) ar;
}

%template(EA_ld) EA<long double>;
%template(rQIEA_ld) rQIEA<long double>;

%clear float ***Q;
*/



/*
%ignore EA::P;
%rename(P) getP();
*/
%extend EA<long double> {
        PyObject* P() {
                int dimensions[2];
                dimensions[0] = self->popsize;
                dimensions[1] = self->chromlen;
                PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(2, dimensions, NPY_LONGDOUBLE);
                long double *ptr = (long double *) ar->data;
                for (int i = 0; i < dimensions[0]; i++) {
                        for (int j = 0; j < dimensions[1]; j++) {
                                *ptr = self->P[i][j];
                                ptr += 1;
                        }
                }
                return (PyObject*) ar;
        }
}

%extend rQIEA<long double> {
    PyObject* Q() {
        int dimensions[3];
        dimensions[0] = self->popsize;
        dimensions[1] = self->chromlen;
        dimensions[2] = 2;
        PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(3, dimensions, NPY_FLOAT);
        float *ptr = (float *) ar->data;
        for (int i = 0; i < dimensions[0]; i++) {
            for (int j = 0; j < dimensions[1]; j++) {
                *ptr = self->Q[i][j][0];
                ptr += 1;
                *ptr = self->Q[i][j][1];
                ptr += 1;
            }
        }
        return (PyObject*) ar;
    }
}

%extend rQIEA<long double> {
    long double getQ(a,b,c) {
            int dimensions[1];
            dimensions[0] = 1;
            PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(1, dimensions, NPY_LONGDOUBLE);
            ((long double*) ar->data)[0] = self->Q[a][b][c];
            return PyArray_Scalar(PyArray_DATA(ar), PyArray_DESCR(ar), (PyObject*) ar);
    }
}

%extend rQIEA<long double> {
    PyObject* bounds() {
        int dimensions[2];
        dimensions[0] = self->chromlen;
        dimensions[1] = 2;
        PyArrayObject *ar =  (PyArrayObject*) PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
        double *ptr = (double *) ar->data;
        for (int i = 0; i < dimensions[0]; i++) {
            *ptr = self->bounds[i][0];
            ptr += 1;
            *ptr = self->bounds[i][1];
            ptr += 1;
        }
        return (PyObject*) ar;
    }

    PyObject* bounds(PyObject *args) {
        PyArrayObject *arr = (PyArrayObject *) args;
        if (NPY_DOUBLE != arr->descr->type_num || arr->nd != 2 || arr->dimensions[0] != self->chromlen || arr->dimensions[1] != 2) {
            PyErr_SetString( PyExc_TypeError,  "bounds() expects (chromlen x 2) numpy array of doubles"); 
            return 0;
        }
        // caution: bounds can be long double
        double *ptr = (double *) arr->data;
        for (int i = 0; i < arr->dimensions[0]; i++) {
            self->bounds[i][0] = *ptr;
            ptr += 1;
            self->bounds[i][1] = *ptr;
            ptr += 1;
        }

        return PyFloat_FromDouble(0);
    }
}



// vim: set ft=c:

