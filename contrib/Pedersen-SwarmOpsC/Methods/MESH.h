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
 *	Optimization method which iterates over all possible
 *	combinations of parameters fitting a mesh of a certain
 *	size. This mesh size is determined by the allowed number
 *	of optimization iterations as follows:
 *	k = pow(numIterations, 1.0/n)
 *	where numIterations is the number of optimization iterations
 *	allowed, n is the dimensionality of the problem to be
 *	optimized, and k is the number of mesh-iterations in each
 *	dimension.
 *
 *	The MESH method is particularly useful for displaying
 *	performance landscapes from meta-optimization, relating
 *	choices of parameters to the performance of an optimization
 *	method.
 *
 * ================================================================ */

#ifndef SO_MESH_H
#define SO_MESH_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */
	/* (None for the MESH method.) */

#define SO_kNumParametersMESH 0
#define SO_kParametersDefaultMESH 0
#define SO_kParametersLowerMESH 0
#define SO_kParametersUpperMESH 0

	/* String containing the name of the optimization method. */
	extern const char SO_kNameMESH[];

	/* Array of strings containing the parameter names of the optimization method. */
#define SO_kParameterNameMESH 0

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_MESH(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_MESH_H */
