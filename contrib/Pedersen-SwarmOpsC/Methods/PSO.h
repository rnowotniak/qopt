/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	PSO
 *
 *	Particle Swarm Optimization (PSO) optimization method originally
 *	due to Eberhart, Shi, Kennedy, etc.
 *
 * ================================================================ */

#ifndef SO_PSO_H
#define SO_PSO_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersPSO 4

	extern const SO_TElm SO_kParametersDefaultPSO[];
	extern const SO_TElm SO_kParametersLowerPSO[];
	extern const SO_TElm SO_kParametersUpperPSO[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNamePSO[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNamePSO[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Number of agents. */
	size_t SO_PSONumAgents(SO_TElm const* param);

	/* Inertia weight. */
	SO_TElm SO_PSOOmega(SO_TElm const* param);

	/* Attraction-weight to particle's own best position. */
	SO_TElm SO_PSOPhi1(SO_TElm const* param);

	/* Attraction-weight to swarm's best position. */
	SO_TElm SO_PSOPhi2(SO_TElm const* param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_PSO(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_PSO_H */
