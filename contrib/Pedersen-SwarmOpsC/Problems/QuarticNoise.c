/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	QuarticNoise
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/QuarticNoise.h>
#include <SwarmOps/Tools/Displace.h>
#include <SwarmOps/Tools/Random.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const char SO_kNameQuarticNoise[] = "QuarticNoise";

/* ---------------------------------------------------------------- */

SO_TFitness SO_QuarticNoise(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	SO_TDim n = c->n;
	int displaceOptimum = c->displaceOptimum;

	SO_TElm value = 0;
	SO_TDim i;

	assert(x);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kQuarticNoiseDisplace);
		SO_TElm elm2 = elm*elm;
		SO_TElm elm4 = elm2*elm2;

		value += ((SO_TElm) (i + 1)) * elm4 + SO_RandUni();
	}

	return value;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_QuarticNoiseGradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	/* Function is discontinuous so gradient does not exist. */
	assert(0);

	return 0;
}

/* ---------------------------------------------------------------- */
