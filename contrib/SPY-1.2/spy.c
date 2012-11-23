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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "random.h"
#include "formula.h"
#include "spy.h"
#include "bk.h"

//global system vars & structures
struct vstruct *v = NULL;           //all per var information
struct clausestruct *clause = NULL; //all per clause information
double *magn = NULL;                //spin magnetization list 
                                    //(for fixing ordering)
int *perm = NULL;                   //permutation, for update ordering
int M = 0;                          //number of clauses
int N = 0;                          //number of variables
int maxlit = 0;                     //maximum clause size
int freespin;                       //number of unfixed variables
double epsilon = EPSILONDEF;        //convergence criterion
int maxconn = 0;                    //maximum connectivity
double y = 2;                      //the temperature 20 = infinity
int update_rate = UPDATE_RATE;      //update rate for Y-updating
double *hsurvey_store, *oldhsurvey_store; //for compute_field
double r = 0.0;                     //fraction of backtrack moves
FILE *out_file;

//auxiliary vars
int *list = NULL;
double *prod = NULL;

//flags & options
double percent = 0;
int fixperstep = 1;
int ndanger = 0;
int generate = 0;
int verbose = 0;
int fields = 0;
int etas = 0;
int stop = 0;
FILE *replay = NULL;
char *load = NULL;
int iterations = ITERATIONS;
int K = 3;
int ygraph = 0;           //evolving optimal pseudo-temperature
int ave_field = 0;        //computing integrated h-pdf's or average fields

extern double expym;


int main(int argc, char ** argv)
{
  int oldfreespin;
  
  //open output file
  if ((out_file = fopen(OUT_FILE, "w")) == NULL) {
    fprintf(stderr, "Cannot open output file\n");
    exit(-1);
  }

  //parse command line and set the verbosity of output messages
  parsecommandline(argc, argv);
  set_verbosity();
  //read or generate a formula
  if (generate) {
    randomformula(K);
  } else if (load) {
    readvarformula(load);
  } else {
    fprintf(out_file, "Error:\n"
	              "you must specify some formula (-l or -n & -a)\n");
    return 0;
  }
  
  //define fixperstep if based on percent
  if (percent)
    fixperstep = N * percent / 100.0;
  
  //allocate mem for dynamics
  initmem();
  
  //write the formula
  writeformula(fopen(FORMULA, "w+"));
  expy(y);    //Compute the exponentials of pseudotemperature 
  
  //pick an initial starting point for the dynamics
  randomize_eta();
  
  //converge and fix fixperstep spins at a time
  fprintf(stderr, "\nSPY Running, the program stores all data to output %s.\n",
	  ( !replay ? "file" : "and replay files" ) );

  while (converge_crossroads()) {
    oldfreespin = freespin;
    
    //decide to backtrack or not 
    to_b_or_not_to_b();
    
    if (fields) print_fields();
    if (etas) print_eta();

    if (stop) return 0;
    
    if (!freespin) {
      close_computation();
      fprintf(out_file, "Finished.\n");
      fclose(out_file);
      return 0;
    }
  }

  fclose(out_file);
  return -1;
}


void set_verbosity(void) 
     //Set appropriately the verbosity level
{
  if (verbose >= 2)
    if (!replay) replay = fopen(REPLAY,"w+");
  if (verbose >= 3)
    fields = 1;
  if (verbose >= 4)
    etas = 1;
  return;
}


int parsecommandline(int argc, char **argv)
     //get command line parameters
{
  double alpha = 0;
  char c;
  generate = 0;
  usrand(1);
  
  while ((c = getopt(argc, argv, "HB:k:N:M:Rn:m:a:s:hf:vy:Yu:%:e:l:FE/i:")) != -1) {
    switch (c) {
    case 'H':
      ave_field = 1;
      break;
    case 'B':
      r = atof(optarg);
      break;
    case 'l':
      load = optarg;
      break;
    case 'R':
      replay = fopen(REPLAY,"w+");
      break;
    case 'N':
    case 'n':
      N = atoi(optarg);
      break;
    case 'M':
    case 'm':
      M = atoi(optarg);
      break;
    case 'a':
      alpha = atof(optarg);
      break;
    case 'e':
      epsilon = atof(optarg);
      break;
    case 's':
      usrand(atoi(optarg));
      srand(atoi(optarg));
      break;
    case '/':
      stop = 1;
      break;
    case 'k':
    case 'K':
      K = atoi(optarg);
      break;
    case 'v':
      verbose++;
      break;
    case 'F':
      fields = 1;
      break;
    case 'E':
      etas = 1;
      break;
    case 'f':
      fixperstep = atoi(optarg);
      break;
    case '%':
      percent = atof(optarg);
      break;
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'y':                         //finite pseudo-temperature
      y = atof(optarg);
      fprintf(out_file, "--------------------------------------------------------\n");
      fprintf(out_file, "Using finite inverse pseudo-temperature y = %1.2f.\n", y);
      fprintf(out_file, "--------------------------------------------------------\n");
      fprintf(out_file, "\n\n");
      break;
    case 'Y':
      ygraph = 1;
      fprintf(out_file, "-------------------------------------------------\n");
      fprintf(out_file, "Using optimized finite inverse pseudo-temperature\n");
      fprintf(out_file, "-------------------------------------------------\n");     
      break;
    case 'u':
      update_rate = atoi(optarg);
      break;
    case 'h':
    case '?':
    default :
      fprintf(stderr, "%s [options]\n"
	      "\n"
	      "  FORMULA\n"
	      "\t-n <numvars>\n"
	      "\t-m <numclauses>\n"
	      "\t-a <alpha>\n"
	      "\t-l <filename>\t reads formula from file\n"
	      "  SOLVING\n"
	      "\t-B <rate> \t backtrack with rate <rate>\n"
	      "\t-f <fixes>\t per step\n"
	      "\t-%% <fixes>\t per step (%%)\n"
	      "\t-e <error>\t criterion for convergence\n"
	      "\t-i <iter>\t maximum number of iterations until convergence\n"
	      "\t-y <y>\t\t set the inverse pseudo-temperature to y\n"
              "\t-Y \t\t run with runtime optimized pseudo-temperature\n"
	      "\t-u <u>\t\t the update rate for Y is set to <u>\n"
	      "\t-H \t\t rank the variables according to the average local field value\n"
	      "  STATS\n"
	      "\t-R \t\t replay file\n"
	      "\t-F \t\t print fields\n"
	      "\t-E \t\t print u-surveys\n"
	      "\t-v \t\t increase verbosity (0 to 4)\n"
	      "  MISC\n"
	      "\t-s <seed>\t (0 = use time, default = 1)\n"
	      "\t-/\t\t stop after first convergence\n"
	      "\t-h\t\t this help\n\nMore help in README file.\n",argv[0]);
      exit(-1);
    }
  }
	
  if (load && !N && !M && !alpha) {
    generate = 0;
  } else if (N && alpha && !M) {
    M = N * alpha;
    generate = 1;
  } else if (M && alpha && !N) {
    N = M / alpha;
    generate = 1;
  } else if (M && N && alpha == 0) {
    generate = 1;
  } else {
    fprintf(stderr, "Error:\n"
	    "you have to specify exactly TWO of -n,-m and -a,\n"
	    "or -l FILE (and then a formula is read from FILE)\n"
            "Parameter -h calls help.\n");
    exit(-1);
  }
  return 0;
}


int order(void const *a, void const *b)
     //order relation for qsort, uses ranking in magn[]
{
  double aux;
  aux = magn[*((int *)b)] - magn[*((int *)a)];
  return aux < 0 ? -1 : (aux > 0 ? 1 : 0);
}


double index_biased(struct weightstruct H)
     //most biased ranking
{
  return fabs(H.p - H.m);
}


double index_pap(struct weightstruct H)
     //most biased with some fuss
{
  return fabs(H.p - H.m) + randreal() * 0.1;
}


double index_para(struct weightstruct H)
     //least paramagnetic ranking
{
  return H.z;
}


double index_frozen(struct weightstruct H)
     //most paramagnetic ranking
{
  return -H.z;
}


double index_best(struct weightstruct H)
     //min(H.p,H.m) ranking
{
  return -(H.p > H.m ? H.m : H.p);
}


int to_b_or_not_to_b(void)
     //Does backtracking or decimation
{  
  if (randreal() < r) {
    build_list(1, index_biased); //index_biased is not used, actually
    unfix_chunk(fixperstep);
  } else {
    build_list(0, index_biased);
    fix_chunk(fixperstep); 
  }
  return 0;
}


int build_list(int backtrack, double (* index)(struct weightstruct))
     //build an ordered list with order *index which is one of index_?
{
  int var, totsites;
  struct weightstruct H;
  double summag;
  double maxmag;
  summag = 0;
  totsites = 0;
  
  if (backtrack) {
    for (var = 1; var <= N; var++) if (v[var].clauselist) if (v[var].spin != 0) {
      if (ave_field)
	H = compute_field(-var, 1);   //compute local fields
      else
	H = compute_field(var, 1);
      list[totsites++] = var;
      magn[var] = ((double) v[var].spin) * (H.p - H.m); // bactrack indexing
    }                                // do not care about paramagnetic spins

  } else {
    for (var = 1; var <= N; var++) if (v[var].clauselist) if (v[var].spin == 0) {    
      if (ave_field)
	H = compute_field(-var, 1);   //compute local fields
      else
	H = compute_field(var, 1);
      list[totsites++] = var;
      magn[var] = (*index)(H);
      maxmag = H.p > H.m ? H.p : H.m;
      summag += maxmag;
    }
  }
  
  qsort(list, totsites, sizeof(int), &order);
  
  if (!backtrack) if ((summag / totsites) < PARAMAGNET) {
    fprintf(out_file, "Paramagnetic state\n");
    fflush(out_file);
    writesolution(fopen(SOLUTION, "w+"));
    close_computation();
    fclose(out_file);
    exit(0);
  }
  return 0;
}


void randomize_eta()
     //pick initial random values
{
  int i,j;
   
  for(i = 0; i < M; i++)
    for(j = 0; j < clause[i].lits; j++) {
      clause[i].literal[j].eta = (clause[i].type == 1) ? 1 : randreal();
    }
}


void initmem()
     //allocate mem (can be called more than once)
{
  free(perm);
  free(list);
  free(magn);
  perm = calloc(M, sizeof(int));
  list = calloc(N + 1, sizeof(int));
  magn = calloc(N + 1, sizeof(double));

  if (!perm || !list || !magn || !div) {
    fprintf(out_file, "Not enough memory for internal structures\n");
    fclose(out_file);
    exit(-1);
  }

  hsurvey_stores(maxconn); //allocate memory for h-surveys
}


void print_fields()
     //print all H (non-cavity) fields
{
  int var;
  struct weightstruct H;
  for (var = 1; var <= N; var++) if (v[var].clauselist) { 
    H = compute_field(var, 1);
    fprintf(out_file, "H(%i)={%f, %f, %f}\n", var, H.p ,H.z, H.m);
  }
}


void print_eta()
     //print all etas
{
  int c,l;
  for(c = 0; c < M; c++) {
    for(l = 0; l < clause[c].lits; l++) {
      fprintf(out_file, "eta(%i, %i) = %f\n", c, l, clause[c].literal[l].eta);
    }
  }
}


int unfix_chunk(int quant)
     //unfix quant spins at a time, taken from list[]
{
  struct weightstruct H;
  int i = 0;
  
  while ((N - freespin) && quant--) {
    while (v[list[i]].spin == 0)
      i++;
    if (ave_field)
      H = compute_field(-list[i], 1);
    else
      H = compute_field(list[i], 1);
    if (replay) {
      if (ave_field)
	fprintf(replay, ">> Unfix %i\t\t\tH = %1.3f\t\t[%i free variables]\n", 
		list[i], -H.p + H.m, freespin + 1);
      else
	fprintf(replay, ">> Unfix %i\t\t\tH = {%1.3f,%1.3f,%1.3f}\t\t[%i free variables]\n", 
		list[i], H.p, H.z, H.m, freespin + 1);
      fflush(replay);
    }

    if (verbose) {
      fprintf(out_file, "[%i free variables]\n", freespin + 1);
      fflush(out_file);
    }

    fix(list[i], 0);  
  }
  return quant;
}


int fix_chunk(int quant)
     //fix quant spins at a time, taken from list[]
{
  int i = 0;
  struct weightstruct H;
  
  while (freespin && quant--) {
    while(v[list[i]].spin)
      i++;
    if (ave_field)
      H = compute_field(-list[i], 1);
    else
      H = compute_field(list[i], 1);
    if(replay) {
      if (ave_field)
	fprintf(replay,">> Fix %i ---> %s\t\tH = %1.3f\t\t[%i free variables]\n", 
		list[i], H.p > H.m ? "-" : "+", -H.p + H.m, freespin - 1);
      else
	fprintf(replay,">> Fix %i ---> %s\t\tH = {%1.3f, %1.3f, %1.3f}\t\t[%i free variables]\n", 
		list[i], H.p > H.m ? "-" : "+", H.p, H.z, H.m, freespin - 1);
      fflush(replay);
    }

    if (verbose) {
      fprintf(out_file, "[%i free variables]\n", freespin - 1);
      fflush(out_file);
    }

    if (fix(list[i], H.p > H.m ? -1 : 1)) {  
      fprintf(out_file, "Bug... trying to fix an already fixed spin!\n");
      fclose(out_file);
      exit(-1);
    }
  }
  return quant;
}


int converge_crossroads()
     //a router
{
  static int schedule = 0;

  if (ygraph) {
    if (schedule % (2 * update_rate) == 0) {
      backup();
    }
    if (schedule % update_rate == 0) {	
      graph_free_energy();
	expy(y);
    }
    schedule++;
  }
  return converge(1);
}






























