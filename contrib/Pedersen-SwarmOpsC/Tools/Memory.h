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
 *	Functions for allocating and freeing memory.
 *
 * ================================================================ */

#ifndef SO_MEMORY_H
#define SO_MEMORY_H

#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Returns an array of the designated size by calling default malloc()
	 * and printing an error-message in case of lack of memory, and then
	 * exiting. You may also provide your own malloc() function by calling
	 * SO_SetMAlloc() below. */
	void *SO_MAlloc(size_t numBytes);

	/* Free the designated memory. */
	void SO_Free(void* memory);

	/*----------------------------------------------------------------*/

	/* Definition of function-pointers for managing memory. */
	typedef void* (*SO_FMAlloc) (size_t numBytes);
	typedef void (*SO_FFree) (void* memory);

	/* Set the malloc and free functions used by SwarmOps. Do this when
	 * the default are not suitable for your needs. */
	void SO_SetMAlloc(SO_FMAlloc fMAlloc, SO_FFree fFree);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_MEMORY_H */
