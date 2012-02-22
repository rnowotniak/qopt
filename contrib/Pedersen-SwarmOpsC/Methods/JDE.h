/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	JDE
 *
 *	Differential Evolution (DE) optimization method originally
 *	due to Storner and Price. jDE variant due to Brest et al.
 *	This variant claims to be 'self-adaptive' in that it
 *	claims to eliminate the need to choose two parameters
 *	of the original DE, but in reality it introduces an additional
 *	6 parameters, so the jDE variant now has 9 parameters instead
 *	of just 3 of the original DE.
 *
 *	Use SO_kJDECrossover to toggle between different
 *	DE crossover variants.
 *
 * ================================================================ */

#ifndef SO_JDE_H
#define SO_JDE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Methods/DEEngine.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */
	/* Set DE configuration to be used (see DEEngine.h) */

#define SO_kJDECrossover SO_kDECrossoverBest1BinSimple

	/* ---------------------------------------------------------------- */
	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersJDE 9

	extern const SO_TElm SO_kParametersDefaultJDE[];
	extern const SO_TElm SO_kParametersLowerJDE[];
	extern const SO_TElm SO_kParametersUpperJDE[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameJDE[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameJDE[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Number of agents. */
	size_t SO_JDENumAgents(SO_TElm const* param);

	/* JDE parameter: FInit */
	SO_TElm SO_JDEFInit(SO_TElm const* param);

	/* JDE parameter: Fl */
	SO_TElm SO_JDEFl(SO_TElm const* param);

	/* JDE parameter: Fu */
	SO_TElm SO_JDEFu(SO_TElm const* param);

	/* JDE parameter: TauF (or Tau1) */
	SO_TElm SO_JDETauF(SO_TElm const* param);

	/* JDE parameter: CRInit */
	SO_TElm SO_JDECRInit(SO_TElm const* param);

	/* JDE parameter: CRl */
	SO_TElm SO_JDECRl(SO_TElm const* param);

	/* JDE parameter: CRu */
	SO_TElm SO_JDECRu(SO_TElm const* param);

	/* JDE parameter: TauCR (or Tau2) */
	SO_TElm SO_JDETauCR(SO_TElm const* param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_JDE(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_JDE_H */
