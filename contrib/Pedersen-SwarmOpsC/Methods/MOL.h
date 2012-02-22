/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MOL
 *
 *	Many Optimizing Liaisons (MOL) optimization method devised
 *	as a simplification to the PSO method originally due to
 *	Eberhart et al. The MOL method does not have any attraction
 *	to the particle's own best known position, and the algorithm
 *	also makes use of random selection of which particle to
 *	update instead of iterating over the entire swarm.
 *
 * ================================================================ */

#ifndef SO_MOL_H
#define SO_MOL_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersMOL 3

	extern const SO_TElm SO_kParametersDefaultMOL[];
	extern const SO_TElm SO_kParametersLowerMOL[];
	extern const SO_TElm SO_kParametersUpperMOL[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameMOL[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameMOL[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Number of agents. */
	size_t SO_MOLNumAgents(SO_TElm const* param);

	/* Inertia weight. */
	SO_TElm SO_MOLOmega(SO_TElm const* param);

	/* Attraction-weight to swarm's best know position. */
	SO_TElm SO_MOLPhi(SO_TElm const* param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_MOL(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_MOL_H */
