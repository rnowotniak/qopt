/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MetaSolution
 *
 *	Hold the best found parameters from meta-optimization,
 *	and also hold the best found solution to the base-level
 *	optimization problem.
 *
 * ================================================================ */

#ifndef SO_METASOLUTION_H
#define SO_METASOLUTION_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Statistics/Solution.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The struct holding the best found parameters from meta-optimization,
	 * as well as the best found solution to the actual base problem. */
	struct SO_MetaSolution
	{
		struct SO_Solution parameters;			/* Best found parameters. */
		struct SO_Solution problem;				/* Best found solution to base-problem. */
	};

	/* Create and return a MetaSolution-struct. Call this after the
	 * meta-optimization runs have completed. */
	struct SO_MetaSolution SO_MakeMetaSolution(struct SO_MethodContext *metaMethodContext, struct SO_MethodContext *methodContext);

	/* Free the memory contents of the MetaSolution-struct. */
	void SO_FreeMetaSolution(struct SO_MetaSolution *s);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_METASOLUTION_H */
