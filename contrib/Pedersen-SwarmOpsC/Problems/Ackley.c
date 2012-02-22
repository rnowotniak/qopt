/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Ackley
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Ackley.h>
#include <SwarmOps/Tools/Constants.h>
#include <SwarmOps/Tools/Displace.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const char SO_kNameAckley[] = "Ackley";

/* ---------------------------------------------------------------- */

/* A helper-function used in both the fitness- and gradient-functions. */

SO_TFitness SO_SqrtSum (SO_TElm const* x, size_t n, const int displaceOptimum)
{
	SO_TElm sum=0;
	size_t i;

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kAckleyDisplace);
		sum += elm*elm;
	}

	return sqrt(((SO_TFitness) sum)/n);
}

/* ---------------------------------------------------------------- */

/* A helper-function used in both the fitness- and gradient-functions. */

SO_TFitness SO_CosSum (SO_TElm const* x, size_t n, const int displaceOptimum)
{
	SO_TElm sum=0;
	size_t i;

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kAckleyDisplace);
		sum += cos(SO_kPi2*elm);
	}

	return exp(((SO_TFitness) sum)/n);
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Ackley(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	const SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;
	SO_TElm fitness;

	assert(x);
	assert(n>=0);

	fitness = SO_kE + 20 - 20*exp(-0.2 * SO_SqrtSum(x, n, displaceOptimum)) - SO_CosSum(x, n, displaceOptimum);

	/* Rounding errors may cause negative fitnesses to occur even
	 * though the mathematical global minimum has fitness zero.
	 * Ensure this still works with meta-optimization which
	 * requires non-negative fitnesses. */
	if (fitness < 0)
	{
		fitness = 0;
	}

	return fitness;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_AckleyGradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;

	SO_TElm sqrtSum, cosSum;
	const SO_TElm kDimRec = 1.0/n;

	SO_TDim i;

	assert(x);
	assert(n>=0);

	sqrtSum = SO_SqrtSum(x, n, displaceOptimum);
	cosSum = SO_CosSum(x, n, displaceOptimum);

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kAckleyDisplace);

		v[i] = 4*kDimRec*exp(-0.2*sqrtSum)*elm/sqrtSum
			+ cosSum * sin(SO_kPi2*elm) * SO_kPi2 * kDimRec;
	}

	return 0;
}

/* ---------------------------------------------------------------- */
