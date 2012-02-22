/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Printer
 *
 *	Provides an intermediate layer between an optimization method
 *	and problem, which prints the solution vector and its computed
 *	fitness value.
 *
 * ================================================================ */

#ifndef SO_PRINTER_H
#define SO_PRINTER_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The context that holds the data needed for the printer to call
	 * thru to the actual fitness function to be optimized. */
	struct SO_PrinterContext
	{
		SO_FProblem f;					/* The fitness function to be optimized. */
		void *fContext;					/* The problem's context. */
		SO_TDim fDim;					/* Dimensionality of problem. */
	};

	/* Create and return the printer-context. */
	struct SO_PrinterContext SO_MakePrinterContext
		(SO_FProblem f,
		void *fContext,
		SO_TDim fDim);

	/* Free the memory allocated by SO_MakePrinterContext(), but not
	 * memory that was allocated elsewhere, such as fContext. */
	void SO_FreePrinterContext(struct SO_PrinterContext *c);

	/* ---------------------------------------------------------------- */

	/* The function to be called by an optimization method instead of
	 * the actual fitness function to be optimized. */
	SO_TFitness SO_Printer(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_PRINTER_H */
