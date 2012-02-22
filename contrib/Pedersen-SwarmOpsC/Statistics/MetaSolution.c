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
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Statistics/MetaSolution.h>
#include <SwarmOps/Tools/Vector.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_MetaSolution SO_MakeMetaSolution(struct SO_MethodContext *metaMethodContext, struct SO_MethodContext *methodContext)
{
	struct SO_MetaSolution s;

	s.parameters = SO_MakeSolution(metaMethodContext);
	s.problem = SO_MakeSolution(methodContext);

	return s;
}

/* ---------------------------------------------------------------- */

void SO_FreeMetaSolution(struct SO_MetaSolution *s)
{
	SO_FreeSolution(&s->parameters);
	SO_FreeSolution(&s->problem);
}

/* ---------------------------------------------------------------- */
