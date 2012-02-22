/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	RND
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/RND.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

/* ---------------------------------------------------------------- */

/* The RND method has no parameters. */

const char SO_kNameRND[] = "RND";

/* ---------------------------------------------------------------- */

SO_TFitness SO_RND(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_MethodContext *c = (struct SO_MethodContext*) context;

	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;							/* Fitness function. */
	void *fContext = c->fContext;					/* Context for fitness function. */
	SO_TDim n = c->fDim;							/* Dimensionality of problem. */
	SO_TElm const* lowerInit = c->lowerInit;		/* Lower initialization boundary. */
	SO_TElm const* upperInit = c->upperInit;		/* Upper initialization boundary. */
	SO_TElm const* lowerBound = c->lowerBound;		/* Lower search-space boundary. */
	SO_TElm const* upperBound = c->upperBound;		/* Upper search-space boundary. */
	size_t numIterations = c->numIterations;		/* Number of iterations to perform. */
	SO_TElm *g = c->g;								/* Best found position for this run. */
	SO_TFitness *gFitness = &(c->gFitness);			/* Fitness for best found position. */

	/* Allocate agent position. */
	SO_TElm *x = SO_NewVector(n);

	/* Iteration variable. */
	size_t i;

	/* Fitness variable. */
	SO_TFitness fitness;

	/* Initialize best fitness for this run. */
	*gFitness = SO_kFitnessMax;

	for (i=0; i<numIterations; i++)
	{
		/* Sample from entire search-space. */
		SO_InitUniform(x, n, lowerBound, upperBound);

		/* Compute new fitness. */
		fitness = f(x, fContext, *gFitness);

		/* Update best-known position and fitness. */
		if (fitness < *gFitness)
		{
			/* Update best-known position. */
			SO_CopyVector(g, x, n);

			/* Update best-known fitness. */
			*gFitness = fitness;
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, i, *gFitness);
	}

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, g, *gFitness);

	/* Delete agent's position vector. */
	SO_FreeVector(x);

	/* Return best-found fitness. */
	return *gFitness;
}

/* ---------------------------------------------------------------- */
