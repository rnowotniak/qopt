/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Step
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Step.h>
#include <SwarmOps/Tools/Displace.h>
#include <assert.h>
#include <math.h>

/* ---------------------------------------------------------------- */

const char SO_kNameStep[] = "Step";

/* ---------------------------------------------------------------- */

SO_TFitness SO_Step(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
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
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kStepDisplace);
		SO_TElm roundedElm = (SO_TElm) (int) (elm + 0.5);

		value += roundedElm * roundedElm;
	}

	return value;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_StepGradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	/* Function is discontinuous so gradient does not exist. */
	assert(0);

	return 0;
}

/* ---------------------------------------------------------------- */
