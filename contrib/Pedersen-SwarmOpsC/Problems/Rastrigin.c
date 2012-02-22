/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Rastrigin
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Rastrigin.h>
#include <SwarmOps/Tools/Constants.h>
#include <SwarmOps/Tools/Displace.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const char SO_kNameRastrigin[] = "Rastrigin";

/* ---------------------------------------------------------------- */

SO_TFitness SO_Rastrigin(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	const SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;

	SO_TElm value = 0;
	SO_TDim i;

	assert(x);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kRastriginDisplace);
		value += elm*elm + 10 - 10*cos(SO_kPi2*elm);
	}

	return value;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_RastriginGradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	const SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;

	SO_TDim i;

	assert(x);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kRastriginDisplace);
		v[i] = 2*elm + SO_kPi2 * 10 * sin(SO_kPi2*elm);
	}

	return 0;
}

/* ---------------------------------------------------------------- */
