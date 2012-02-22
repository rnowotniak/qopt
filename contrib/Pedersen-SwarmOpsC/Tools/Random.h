/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Random
 *
 *	Functions for creating pseudo-random numbers.
 *	By default these map directly to RandomOps functions.
 *	If you need another PRNG then change the source-file.
 *
 * ================================================================ */

#ifndef SO_RANDOM_H
#define SO_RANDOM_H

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Return a uniform random number in the inclusive range [0,1] */
	double SO_RandUni();

	/* Return a uniform random number in the inclusive range [-1,1] */
	double SO_RandBi();

	/* Return a uniform random number in the range (lower, upper) */
	double SO_RandBetween(const double lower, const double upper);

	/* Return a Gaussian (or normal) distributed random number. */
	double SO_RandGauss(double mean, double deviation);

	/* Return a random number from {0, .., n-1} with equal probability. */
	size_t SO_RandIndex(size_t n);

	/* Return two distinct numbers from {0, .., n-1} with equal probability. */
	void SO_RandIndex2(size_t n, size_t *i1, size_t *i2);

	/* Return a random boolean. */
	int SO_RandBool();

	/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_RANDOM_H */
