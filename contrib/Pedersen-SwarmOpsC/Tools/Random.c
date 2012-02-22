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
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <RandomOps/random.h>

/* ---------------------------------------------------------------- */

double SO_RandUni()
{
	return RO_RandUni();
}

/* ---------------------------------------------------------------- */

double SO_RandBi()
{
	return RO_RandBi();
}

/* ---------------------------------------------------------------- */

double SO_RandBetween(const double lower, const double upper)
{
	return RO_RandBetween(lower, upper);
}

/* ---------------------------------------------------------------- */

double SO_RandGauss(double mean, double deviation)
{
	return RO_RandGauss(mean, deviation);
}

/* ---------------------------------------------------------------- */

size_t SO_RandIndex(size_t n)
{
	return RO_RandIndex(n);
}

/* ---------------------------------------------------------------- */

void SO_RandIndex2(size_t n, size_t *i1, size_t *i2)
{
	RO_RandIndex2(n, i1, i2);
}

/* ---------------------------------------------------------------- */

int SO_RandBool()
{
	return RO_RandBool();
}

/* ---------------------------------------------------------------- */
