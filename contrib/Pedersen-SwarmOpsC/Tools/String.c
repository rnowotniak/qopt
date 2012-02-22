/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	String
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/String.h>
#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Tools/Memory.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

char *SO_CopyString(const char *s)
{
	char *sCopy;

	if (s != 0)
	{
		/* Find length of original string. */
		size_t size = strlen(s)+1;

		/* Allocate new memory to fit a copy of the string.
		 * Allocate 1 extra byte for null-termination. */
		sCopy = (char*) SO_MAlloc(sizeof(char)*size);

		/* Perform copying of the string. */
		strcpy(sCopy, s);
	}
	else
	{
		/* In case of nil pointer as input, return nil point. */
		sCopy = 0;
	}

	return sCopy;
}

/* ---------------------------------------------------------------- */

void SO_FreeString(char *s)
{
	SO_Free(s);
}

/* ---------------------------------------------------------------- */
