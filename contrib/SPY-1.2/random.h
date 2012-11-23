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

void usrand (unsigned seed);
unsigned urand0 (void);
unsigned urand (void);
int randint(int upto);
double randreal();

#define MAX_URAND 0xFFFFFFFFL
