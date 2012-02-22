/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Penalized1
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Penalized1.h>
#include <SwarmOps/Tools/Constants.h>
#include <assert.h>
#include <math.h>

/* ---------------------------------------------------------------- */

const char SO_kNamePenalized1[] = "Penalized1";

/* ---------------------------------------------------------------- */

SO_TElm SO_PenalizedU(const SO_TElm x, const SO_TElm a, const SO_TElm k, const SO_TElm m)
{
	SO_TElm value;

	if (x<-a)
	{
		value = k * pow(-x-a, m);
	}
	else if (x>a)
	{
		value = k * pow(x-a, m);
	}
	else
	{
		value = 0;
	}

	return value;
}

/* ---------------------------------------------------------------- */

SO_TElm SO_Penalized1GetY(const SO_TElm *x, size_t i)
{
	SO_TElm elm = x[i];

	return 1 + 0.25 * (elm + 1);
}

/* ---------------------------------------------------------------- */

SO_TElm SO_Penalized1GetSinY(const SO_TElm *x, size_t i)
{
	SO_TElm elmY = SO_Penalized1GetY(x, i);
	SO_TElm elmSinY = sin(SO_kPi * elmY);

	return 10 * elmSinY * elmSinY;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Penalized1(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
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
	value = SO_Penalized1GetSinY(x, 0);

	for (i=0; i<n-1; i++)
	{
		SO_TElm elmY = SO_Penalized1GetY(x, i);
		SO_TElm elmYMinusOne = elmY-1;

		value += (elmYMinusOne*elmYMinusOne) * (1 + SO_Penalized1GetSinY(x, i+1));
	}

	/* Add last y-term. */
	{
		SO_TElm elmY = SO_Penalized1GetY(x, n-1);
		SO_TElm elmYMinusOne = elmY-1;

		value += elmYMinusOne * elmYMinusOne;
	}

	/* Compute penalty. */
	for (i=0; i<n; i++)
	{
		SO_TElm elm = x[i];

		penalty += SO_PenalizedU(elm, 10.0, 100.0, 4.0);
	}

	return SO_kPi * value / n + penalty;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_Penalized1Gradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	/* Gradient not implemented. */
	assert(0);

	return 0;
}

/* ---------------------------------------------------------------- */
