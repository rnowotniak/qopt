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

//function prototypes

void hsurvey_stores(int maxconn);
int converge(int write_flag);
void expy(double y);
double iterate(void);
double update_eta(int cl);
struct weightstruct compute_field(int var, int local_flag);
double compute_free_energy(void);
void graph_free_energy(void);
void rtbis(double y1, double y2);
void backup(void);

//macros
#define test_array(array) if ((array) == NULL) {fprintf(out_file, "Error: Not enough memory for internal structures."); exit(-1);}
#define converge_with_(y) (fprintf(out_file, "Trying y = %1.2e.\n", (y)), randomize_eta(), expy(y), converge(0))

