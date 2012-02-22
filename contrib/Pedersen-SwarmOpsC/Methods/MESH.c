/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MESH
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/MESH.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The MESH method has no parameters. */

const char SO_kNameMESH[] = "MESH";

/* ---------------------------------------------------------------- */

/* Helper function for recursive traversal of the mesh in a depth-first order. */
void SO_MESHRecursive(	struct SO_MethodContext *c,			/* Context for MESH method. */
						 size_t curDim,						/* Current dimension processed. */
						 const size_t kNumIterationsPerDim,	/* Number of mesh iterations per dimension. */
						 const SO_TElm *kDelta,				/* Distance between points in mesh. */
						 SO_TElm *x)						/* Current mesh point. */
{
	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;							/* Fitness function. */
	void *fContext = c->fContext;					/* Context for fitness function. */
	SO_TDim n = c->fDim;							/* Dimensionality of problem. */
	SO_TElm const* lowerBound = c->lowerBound;		/* Lower search-space boundary. */
	SO_TElm const* upperBound = c->upperBound;		/* Upper search-space boundary. */
	SO_TElm *g = c->g;								/* Best found position for this run. */
	SO_TFitness *gFitness = &(c->gFitness);			/* Fitness for best found position. */

	/* Iteration variable. */
	size_t i;

	/* Iterate over all mesh-entries for current dimension. */
	for (i=0; i<kNumIterationsPerDim; i++)
	{
		assert(curDim>=0 && curDim<n);

		/* Update mesh position for current dimension. */
		x[curDim] = lowerBound[curDim] + kDelta[curDim] * i;

		/* Bound mesh position for current dimension. */
		SO_BoundOne(x, curDim, lowerBound, upperBound);

		/* Eithe recurse or compute fitness for mesh position. */
		if (curDim < n-1)
		{
			/* Recurse for next dimension. */
			SO_MESHRecursive(c, curDim+1, kNumIterationsPerDim, kDelta, x);
		}
		else
		{
			/* Compute fitness for current mesh position. */
			SO_TFitness fitness = f(x, fContext, SO_kFitnessMax);

			/* Update best position and fitness found in this run. */
			if (fitness<*gFitness)
			{
				/* Update this run's best known position. */
				SO_CopyVector(g, x, n);

				/* Update this run's best know fitness. */
				*gFitness = fitness;
			}
		}
	}
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_MESH(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_MethodContext *c = (struct SO_MethodContext*) context;

	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;							/* Fitness function. */
	void *fContext = c->fContext;					/* Context for fitness function. */
	SO_TDim n = c->fDim;							/* Dimensionality of problem. */
	SO_TElm const* lowerBound = c->lowerBound;		/* Lower search-space boundary. */
	SO_TElm const* upperBound = c->upperBound;		/* Upper search-space boundary. */
	size_t numIterations = c->numIterations;		/* Number of iterations to perform. */
	SO_TElm *g = c->g;								/* Best found position for this run. */
	SO_TFitness *gFitness = &(c->gFitness);			/* Fitness for best found position. */

	/* Allocate mesh position and mesh-incremental values. */
	SO_TElm* x = SO_NewVector(n);					/* Mesh position. */
	SO_TElm* delta = SO_NewVector(n);				/* Mesh incremental values. */

	/* Compute the number of iterations per dimension in the mesh. Rounded to nearest. */
	const size_t kNumIterationsPerDim = (size_t) (pow(numIterations, 1.0/n) + 0.5);

	/* Iteration variable. */
	size_t i;

	assert(kNumIterationsPerDim>1);

	/* Initialize best found fitness. */
	*gFitness = SO_kFitnessMax;

	/* Initialize mesh position to the lower boundary. */
	SO_CopyVector(x, lowerBound, n);

	/* Compute mesh incremental values for all dimensions. */
	for (i=0; i<n; i++)
	{
		delta[i] = (upperBound[i]-lowerBound[i]) / (kNumIterationsPerDim-1);
	}

	/* Start recursive traversal of mesh. */
	SO_MESHRecursive(c, 0, kNumIterationsPerDim, delta, x);

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, g, *gFitness);

	/* Delete vectors for the mesh position and mesh-incremental values. */
	SO_FreeVector(x);
	SO_FreeVector(delta);

	return *gFitness;
}

/* ---------------------------------------------------------------- */
