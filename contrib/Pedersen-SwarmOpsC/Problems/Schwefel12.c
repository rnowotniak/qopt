/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Schwefel12
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Schwefel12.h>
#include <SwarmOps/Tools/Displace.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const char SO_kNameSchwefel12[] = "Schwefel1-2";

/* ---------------------------------------------------------------- */

SO_TFitness SO_Schwefel12(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	SO_TDim n = c->n;
	int displaceOptimum = c->displaceOptimum;

	SO_TElm value = 0;
	SO_TDim i, j;

	assert(x);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		SO_TElm sum = 0;

		for (j=0; j<=i; j++)
		{
			SO_TElm elm = SO_Displace(x, j, displaceOptimum, SO_kSchwefel12Displace);

			sum += elm;
		}

		value += sum*sum;
	}

	return value;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_Schwefel12Gradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	/* Gradient not implemented. */
	assert(0);

	return 0;
}

/* ---------------------------------------------------------------- */
