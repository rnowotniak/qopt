/*
    Copyright 2004 Demian Battaglia, Alfredo Braunstein, Michal Kolar,
    Michele Leone, Marc Mezard, Martin Weigt and Riccardo Zecchina

    This file is part of SPY (Survey Propagation with finite Y).

    SPY is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    SPY is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SP; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/


#include <time.h>
#include "random.h"

/*
	urand(), urand0() return uniformly distributed unsigned random ints
		all available bits are random, e.g. 32 bits on many platforms
	usrand(seed) initializes the generator
		a seed of 0 uses the current time as seed
	
	urand0() is the additive number generator "Program A" on p.27 of Knuth v.2
	urand() is urand0() randomized by shuffling, "Algorithm B" on p.32 of Knuth v.2
	urand0() is one of the fastest known generators that passes randomness tests
	urand() is somewhat slower, and is presumably better
*/

static unsigned rand_x[56], rand_y[256], rand_z;
static int rand_j, rand_k;

void usrand (unsigned seed)
{
	int j;

	rand_x[1] = 1;
	if (seed) rand_x[2] = seed;
	else rand_x[2] = time (NULL);
	for (j=3; j<56; ++j) rand_x[j] = rand_x[j-1] + rand_x[j-2];
	
	rand_j = 24;
	rand_k = 55;
	for (j=255; j>=0; --j) urand0 ();
	for (j=255; j>=0; --j) rand_y[j] = urand0 ();
	rand_z = urand0 ();
}

unsigned urand0 (void)
{
	if (--rand_j == 0) rand_j = 55;
	if (--rand_k == 0) rand_k = 55;
	return rand_x[rand_k] += rand_x[rand_j];
}

unsigned urand (void)
{
	int j;
	
	j =  rand_z >> 24;
	rand_z = rand_y[j];
	if (--rand_j == 0) rand_j = 55;
	if (--rand_k == 0) rand_k = 55;
	rand_y[j] = rand_x[rand_k] += rand_x[rand_j];
	return rand_z;
}

int randint(int upto)
//gives a random integer uniformly in [0..upto-1]
{
//	return((float)upto*(float)rand())/(RAND_MAX+1.0);
	return(((double)upto*(double)urand())/((double)MAX_URAND+1.0));
}

double randreal()
//gives a random real uniformly in (0,1)
{
//	return (rand()/(RAND_MAX+1.0));
	return (urand()/(double)((double)MAX_URAND+1.0));	
}
