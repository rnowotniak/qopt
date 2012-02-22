/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Rosenbrock
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Rosenbrock.h>
#include <SwarmOps/Tools/Displace.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const char SO_kNameRosenbrock[] = "Rosenbrock";

/* ---------------------------------------------------------------- */

SO_TFitness SO_Rosenbrock(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	const SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;

	SO_TElm value = 0;
	SO_TDim i;

	assert(x);
	assert(n>=0);

	for (i=0; i<n-1; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kRosenbrockDisplace);
		SO_TElm nextElm = SO_Displace(x, i+1, displaceOptimum, SO_kRosenbrockDisplace);

		SO_TElm minusOne = elm-1;
		SO_TElm nextMinusSqr = nextElm - elm*elm;
			
		value += 100*nextMinusSqr*nextMinusSqr + minusOne*minusOne;
	}

	return value;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_RosenbrockGradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	const SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;

	SO_TDim i;

	assert(x);
	assert(n>=2);

	for (i=0; i<n-1; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kRosenbrockDisplace);
		SO_TElm nextElm = SO_Displace(x, i+1, displaceOptimum, SO_kRosenbrockDisplace);

		v[i] = -400 * (nextElm - elm*elm) * elm + 2*(elm-1);
	}

	/* Gradient for the last dimension. */
	{
		SO_TElm elm = SO_Displace(x, n-1, displaceOptimum, SO_kRosenbrockDisplace);
		SO_TElm prevElm = SO_Displace(x, n-2, displaceOptimum, SO_kRosenbrockDisplace);;

		v[n-1] = 200*(elm - prevElm*prevElm);
	}

	return 0;
}

/* ---------------------------------------------------------------- */
