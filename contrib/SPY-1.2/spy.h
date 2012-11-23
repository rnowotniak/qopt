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

//files
#define OUT_FILE "output.tmp.out"
#define REPLAY "replay.tmp.out"
#define FORMULA "formula.tmp.cnf"
#define SUBFORMULA "subformula.tmp.cnf"
#define NOCONVERGENCE "noconvergence.tmp.cnf"
#define SPSOL "spsol.tmp.lst"
#define SOLUTION "fixed_spins.tmp.lst"

//default parameters
#define   ITERATIONS 1000  // max # iterations before giving up
#define   EPSILONDEF 0.0001  // convergence check
#define   EPS (1.0e-9)     // zero as we see it

//fixing strategy (see function fix_*)
#define   ME 0.1
#define   MEBIL 0.2
#define   MENOBIL 0.001
#define   MEZERO 0.1
#define   PARAMAGNET 0.01

//finite y mode
#define MIN_Y  0.6               //search interval for pseudotemperature 
#define MAX_Y  6
#define STEP_Y 0.05
#define ACCURACY_Y 0.05          //accuracy of pseudotemperature
#define PHASE_TRANS 0.01         //
#define UPDATE_RATE 25           //how often to update (the less the more often)
#define UPDATE_ITERATIONS 250    //number of iterations before giving up
                                 //pseudotemperature-updating procedure
#define FREE_ENERGY_FILE "free_energy.tmp" 
                                 //where to store free energy data

//function list
int parsecommandline(int argc, char **argv);
void randomize_eta();
void initmem();
void print_fields();
void print_eta();
int converge();
int converge_crossroads(void);
int fix_biased();
int fix_best();
int fix_balanced();
int fix(int var, int spin); 
int fix_chunk(int quant);
int build_list(int to_b, double (* index)(struct weightstruct));
double index_biased(struct weightstruct H);
double index_best(struct weightstruct H);
double index_frozen(struct weightstruct H);
double index_para(struct weightstruct H);
double index_pap(struct weightstruct H);
int order(void const *a, void const *b);
int to_b_or_not_to_b(void);
int unfix_chunk(int quant);
void set_verbosity(void);

