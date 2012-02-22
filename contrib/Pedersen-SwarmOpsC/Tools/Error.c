/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Error
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The error messages. */

const char SO_kErrMsgMemory[] = "Out of memory.";

/* ---------------------------------------------------------------- */

void SO_Error(const char *msg, int no)
{
	assert(msg);
	assert(no != 0);

	printf("Error: %s\nError number: %i", msg, no);

	exit(no);
}

/* ---------------------------------------------------------------- */
