/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Sample
 *
 *	Functions for making random samples.
 *
 * ================================================================ */

#ifndef SO_SAMPLE_H
#define SO_SAMPLE_H

#include <SwarmOps/Tools/Types.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* First bound the sampling range [x-range, x+range] using lowerBound
	 * and upperBound respectively, and then pick a uniform random sample
	 * from the bounded range. This avoids samples being boundary points. */
	SO_TElm SO_SampleBoundedOne(const SO_TElm x, const SO_TElm range, const SO_TElm lowerBound, const SO_TElm upperBound);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_SAMPLE_H */
