/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Penalized2
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Penalized2.h>
#include <SwarmOps/Problems/Penalized1.h> /* SO_PenalizedU() function */
#include <SwarmOps/Tools/Constants.h>
#include <assert.h>
#include <math.h>

/* ---------------------------------------------------------------- */

const char SO_kNamePenalized2[] = "Penalized2";

/* ---------------------------------------------------------------- */

SO_TElm SO_Penalized2GetSinX(const SO_TElm *x, size_t i, const SO_TElm factor)
{
	SO_TElm elm = x[i];
	SO_TElm elmSin = sin(factor * SO_kPi * elm);

	return elmSin * elmSin;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Penalized2(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	SO_TDim n = c->n;
	/* Ignore optimum displacement because of penalty function.
	 * int displaceOptimum = c->displaceOptimum; */

	SO_TElm value = 0;
	SO_TDim i;
	SO_TElm penalty = 0;

	assert(x);
	assert(n>=0);

	/* Compute main fitness value ... */
	value = SO_Penalized2GetSinX(x, 0, 3.0);

	for (i=0; i<n-1; i++)
	{
		SO_TElm elm = x[i];
		SO_TElm elmMinusOne = elm-1;
		SO_TElm elmSin = SO_Penalized2GetSinX(x, i+1, 3.0);

		value += (elmMinusOne*elmMinusOne) * (1 + elmSin);
	}

	/* Add last term. */
	{
		SO_TElm elm = x[n-1];
		SO_TElm elmMinusOne = elm-1;
		SO_TElm elmSin = SO_Penalized2GetSinX(x, n-1, 2.0);

		value += elmMinusOne * elmMinusOne * (1 + elmSin);
	}

	/* Compute penalty. */
	for (i=0; i<n; i++)
	{
		SO_TElm elm = x[i];

		penalty += SO_PenalizedU(elm, 5.0, 100.0, 4.0);
	}

	return 0.1 * value + penalty;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_Penalized2Gradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	/* Gradient not implemented. */
	assert(0);

	return 0;
}

/* ---------------------------------------------------------------- */
