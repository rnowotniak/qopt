/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Memory
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Memory.h>
#include <SwarmOps/Tools/Error.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

void *SO_DoMAlloc(size_t numBytes)
{
	void *newMemory = malloc(numBytes);

	if (!newMemory)
	{
		SO_Error(SO_kErrMsgMemory, SO_kErrNoMemory);
	}

	assert(newMemory);

	return newMemory;
}

/* ---------------------------------------------------------------- */

void SO_DoFree(void* memory)
{
	free(memory);
}

/* ---------------------------------------------------------------- */

/* Initialize the functions to SO_DoMAlloc() and SO_DoFree. */
SO_FMAlloc SO_gFMAlloc = SO_DoMAlloc;
SO_FFree SO_gFFree = SO_DoFree;

/* ---------------------------------------------------------------- */

void SO_SetMAlloc(SO_FMAlloc fMAlloc, SO_FFree fFree)
{
	SO_gFMAlloc = fMAlloc;
	SO_gFFree = fFree;
}

/* ---------------------------------------------------------------- */

void *SO_MAlloc(size_t numBytes)
{
	return SO_gFMAlloc(numBytes);
}

/* ---------------------------------------------------------------- */

void SO_Free(void* memory)
{
	SO_gFFree(memory);
}

/* ---------------------------------------------------------------- */
