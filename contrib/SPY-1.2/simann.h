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

//global system vars & structures
struct vstruct *v=NULL; //all per var information
struct clausestruct *clause=NULL; //all per clause information
int M=0; //number of clauses
int N=0; //number of variables
int *ncl=NULL; //clause size histogram
int maxlit=0; //maximum clause size
int maxconn=0; //maximum connectivity

FILE *af;
FILE *adf;
FILE *sf;

//flags & options
int generate = 0;
int verbose = 0;
int output_sol = 0;
char *load=NULL;
int K = 3;

//schedule variables

int MC_time = 1000;
double init_temp = 0.5;
double final_temp =  0.0;
int therm_time = 2;
int n_statistics = 10;            //number of experiments per run

//other annealing-specific variables

int *spins;
int  *best_spins;                    //vector of variables' spins
double *moment_1, *moment_2, *actual_temp; 
double temperature; //temporary value of temperature
int best_energy;
int n_hits = 0;


char ANNEALING_FILE[50];
char ALLDATA_FILE[50];
char SPINS_FILE[50];

//function prototypes
void annealing(void);
void single_annealing(void);
int configuration_energy(void);
int evaluate_clause(struct clausestruct *cl);
int metropolis(int var);
int parsecommandline(int argc, char **argv);
void read_write_files(void);
void init_spins(void);
void close_annealing(void);
void init_statistics(void);
void randomformula(int K);
int readvarformula(char *filename);
int writeformula(FILE *sink);

//macros
#define test_array(array) if ((array) == NULL) {fprintf(stderr, "Error: Not enough memory for internal structures."); exit(-1);}
