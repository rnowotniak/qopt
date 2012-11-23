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

//to change standard filenames, see function read_write_files

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "random.h"
#include "formula.h"
#include "simann.h"
#include "spy.h"


int main(int argc, char ** argv)
{
  //parse command line and decide file_names
  parsecommandline(argc,argv);
  read_write_files();
  init_spins();
  
  //do annealing
  annealing();
  return 0;
}


void read_write_files(void)
     //open the needed output files
{
  sprintf(ANNEALING_FILE, "L%i-j%i-t%1.3f-R%i.tmp.ann", MC_time, therm_time, 
	  init_temp, n_statistics);
  sprintf(SPINS_FILE, "L%i-j%i-t%1.3f-R%i.tmp.sol", MC_time, therm_time, 
	  init_temp, n_statistics);
  sprintf(ALLDATA_FILE, "L%i-j%i-t%1.3f-R%i.tmp.all", MC_time, therm_time, 
	  init_temp, n_statistics);
  
  //read or generate a formula
  if(generate) {
    randomformula(K);
    //write the formula and the schedule
    writeformula(fopen(FORMULA,"w+"));
  } else if (load) {
    readvarformula(load);
  } else {
    fprintf(stderr, "error: you must specify some formula (-l or -n & -a)\n");
    exit(-1);
  }
  return;
} 


int parsecommandline(int argc, char **argv)
     //get command line parameters
{
  double alpha = 0;
  char c;
  generate = 0;
  usrand(1);
  while((c = getopt(argc, argv,"l:N:n:Mm:a:s:k:K:vL:t:T:hj:J:R:S")) != -1) {
    switch (c) {
    case 'l':
      load=optarg;
      break;
    case 'N':
    case 'n':
      N = atoi(optarg);
      break;
    case 'm':
      M = atoi(optarg);
      break;
    case 'a':
      alpha = atof(optarg);
      break;
    case 's':
      usrand(atoi(optarg));
      srand(atoi(optarg));
      break;
      break;
    case 'k':
    case 'K':
      K=atoi(optarg);
      break;
    case 'v':
      verbose++;
      break;
    case 'L':
      MC_time = atoi(optarg);
      break;
    case 't':
      init_temp = atof(optarg);
      break;
    case 'T':
      final_temp = atof(optarg);
      break;
    case 'j':
    case 'J':
      therm_time = atoi(optarg);
      break;
    case 'R':
      n_statistics = atoi(optarg);
      break;
    case 'S':
      output_sol = 1;
      break;
    case 'h':
    default :
      fprintf(stderr, "%s [options]\n"
	      "\n"
	      "  formula\n"
	      "\t-n <numvars>\n"
	      "\t-m <numclauses>\n"
	      "\t-a <alpha>\n"
	      "\t-K <K> (default = 3)\n"
	      "\t-l <filename>\t reads formula from file\n"
	      "\t-v \t\t increase verbosity\n"
	      "\t-S \t\t output best assignment\n"
	      "\t-s <seed>\t (0=use time, default=1)\n"
	      "\t-R <n>\t\t number of experiments performed during the run\n"
	      "-----------------------------\n"
	      "\tSCHEDULE PARAMETERS\n"
	      "-----------------------------\n"
	      "\t-t\t\t initial temperature\n"
	      "\t-T\t\t final temperature\n"
	      "\t-j\t\t thermalization time\n"
	      "\t-L\t\t number of temperature steps\n"
	      "-----------------------------\n"
	      "\t-h\t\t this help!\n", argv[0]);
      exit(-1);
    }
  }
  
  if(load && !N && !M && !alpha) {
    generate=0;
  } else if(N && alpha && !M) {
    M=N*alpha;
    generate=1;
  } else if(M && alpha && !N) {
    N=M/alpha;
    generate=1;
  } else if(M && N && alpha==0) {
    generate=1;
  } else {
    fprintf(stderr, "error: you have to specify exactly TWO of -n,-m and -a, or -l FILE (and then a formula is read from FILE)\n");
    exit(-1);
  }
  return 0;
}


void init_spins(void)
     //initializes spin array
{
   
  spins = calloc(N + 1, sizeof(int));
  best_spins = calloc(N + 1, sizeof(int));
  test_array(spins);
  test_array(best_spins);
  
  return;
}


int configuration_energy(void)
     //computes configuration energy
{
  int c;
  int energy;
  
  energy = 0;
  for (c = 0; c < M; c++) {
    energy += evaluate_clause(&clause[c]);
  }
  return energy;
}


int evaluate_clause(struct clausestruct *cl)
     //evaluates energy of a single clause
{
  int lits, m;
  
  lits = cl->lits;
  for (m = 0; m < lits; m++) {
    if ( (cl->literal[m].bar) && (spins[cl->literal[m].var] < 0) ) 
      return 0;
    if ( (!(cl->literal[m].bar)) && (spins[cl->literal[m].var] > 0) ) 
      return 0;
  }
  return 1;
}


void init_statistics(void)
     //initializes the statistics vectors needed depending of the type of annealing to be 
     //performed
{
  moment_1 = calloc((MC_time + 1) * therm_time + 1, sizeof(double));
  //This will be the average of energies, taken over different runs
  moment_2 = calloc((MC_time + 1) * therm_time + 1, sizeof(double));
  //This will be the standard deviation of the average energy
  actual_temp = calloc((MC_time + 1) * therm_time + 1, sizeof(double));
  //This will be the temperature at a given time
  
  
  if ((af = fopen(ANNEALING_FILE, "w")) == NULL) {
    fprintf(stderr, "Cannot open the annealing output file\n");
    exit(-1);
  }
  if (verbose)
    if ((adf = fopen(ALLDATA_FILE, "w")) == NULL) {
      fprintf(stderr, "Cannot open the all_data output file\n");
      exit(-1);
    }
  
  return;
  }


void close_annealing(void)
{
  int n;
  
  fclose(af);
  if (verbose >=2 || output_sol) {
    if ((sf = fopen(SPINS_FILE, "w")) == NULL) {
      fprintf(stderr, "Cannot open the spins store file\n");
      exit(-1);
    }
    for (n = 1; n <= N; n++) if (best_spins[n]) 
      fprintf(sf, "%5d\n", (best_spins[n] > 0) ? n : -n);
    fclose(sf);
  }
  if (verbose) 
    fclose(adf);
  
  return;
}

/*------------------------------------------------------------------------------------*/
/*                  HERE BEGINS THE MONTECARLO CORE OF THE PROGRAM!!!!!               */
/*------------------------------------------------------------------------------------*/


void annealing(void)
     //produces statistics over STATISTICs different classical annealing runs 
{
  double stddev = 0;
  int m;
   
  init_statistics();   //initialize the statistics vectors for classical annealing
  
  best_energy = M;

  //Call n_statistics times the classical annealing function
  
  for (m = 0; m < n_statistics; m++) {
    if (verbose){
      fprintf(adf,"\n#%i\n", m + 1);
      fflush(adf);
    }
    single_annealing();
  }
  if (verbose) {
    fprintf(adf, "\n#### DONE! ####\n");
    fflush(adf);
  }
  //Compute average and standard deviation among different runs; write output

  for (m = 0; m < ((MC_time) * therm_time); m++) {
    stddev = sqrt((moment_2[m] - moment_1[m] * moment_1[m]) / ((double) n_statistics));
    fprintf(af, "%d\t%f\t%f\t%f\n", m + 1, actual_temp[m], moment_1[m], stddev);
  }
  fprintf(af, "\n*\n");
  fprintf(af, "MC time\tFinal energy\tFinal STD\tBest energy\tNumber of hits\n");
  fprintf(af, "%d\t%f\t%f\t%d\t\t%d\n", MC_time * therm_time, moment_1[--m], stddev, best_energy, n_hits);
    
  close_annealing();
  return;
}

void single_annealing(void)
     //launch classical annealing experiments
{
  int energy, best_energy_this_run;
  int var, m, n;
  double deltaT;
  
  best_energy_this_run = M;
  
  for (var = 1; var <= N; var++) 
    spins[var] = 2 * randint(2) - 1;
  
  energy = configuration_energy(); //computes configuration energy
  if (verbose) {
    fprintf(adf, "Initial energy = %d\n", energy);   
    fflush(adf);
  }
  deltaT = (init_temp - final_temp) / ((double) MC_time);
  temperature = init_temp;
  
  for (m = 0; m <= MC_time; m++) {
    for (n = 1; n <= therm_time; n++) {
      for (var = 1; var <= N; var++) 
	energy += metropolis(var);
      if (energy == best_energy)
	n_hits++;
      if (energy < best_energy) {
	best_energy = energy;
	n_hits = 1;
      }
      
      if (energy < best_energy_this_run)
	best_energy_this_run = energy;
      if (best_energy_this_run == 0) {
	if (verbose) {
	  fprintf(adf, "SAT assignment found! ===> %d\t%f\t%d\n", m * therm_time + n, temperature, energy);
	  fflush(adf);
	}
	return;
      }
      
      moment_1[m * therm_time + n - 1] 
	+= ((double) energy) / ((double) n_statistics);
      moment_2[m * therm_time + n - 1] 
	+= ((double) energy) * ((double) energy) / ((double) n_statistics); 
      actual_temp[m * therm_time + n - 1] = temperature;
    }
    temperature -= deltaT;
    temperature = (temperature > 0) ? temperature : 0; 
  }
  return;
}


int metropolis(int var)
     //Try to flip one single spin
{
  int m, conn, oldenergy, newenergy, deltaenergy;
  struct clauselist *clauselist;
  
  oldenergy = 0;
  newenergy = 0;
  conn = v[var].clauses;
  
  for (m = 0, clauselist = v[var].clauselist; m < conn; m++, clauselist++)
    if (clauselist->clause->type)
      oldenergy += evaluate_clause(clauselist->clause);
  
  spins[var] = -spins[var];
  
  for (m = 0, clauselist = v[var].clauselist; m < conn; m++, clauselist++)
    newenergy += evaluate_clause(clauselist->clause);
  
  deltaenergy = newenergy - oldenergy;

  if ( (deltaenergy <= 0) || (randreal() < exp(-((double) deltaenergy) / temperature)) )
    return deltaenergy;
  
  spins[var] = -spins[var];
  return 0;
}


int readvarformula(char *filename)
     //read a dimacs cnf formula from a file (not a pipe)
{
  FILE * source;
  struct literalstruct *allliterals = NULL;
  struct clauselist *allclauses;
  int aux, num, cl = 0, lit = 0, var, literals = 0;
  int unused;
  maxconn = 0;
  source = fopen(filename,"r");
  if (!source) {
    fprintf(stderr, "Error in formula file!\n Unable to read %s.\n", filename);
    fclose(stderr);
    exit(-1);
  }
  fprintf(stderr, "reading variable clause-size formula %s ", filename);
  //skip comments
  while ((aux = getc(source)) == 'c') {
    while(getc(source) != '\n');
  }
  ungetc(aux, source);
  fprintf(stderr, ".");
  
  //read p line
  fscanf(source, "p cnf %i %i", &N, &M);
  v = calloc(N + 1, sizeof(struct vstruct));
  clause = calloc(M, sizeof(struct clausestruct));
  //first pass for counting
  fprintf(stderr, ".");
  
  while (fscanf(source, "%i ", &num) == 1) {
    if (!num) {
      if (cl == M) {
	fprintf(stderr, "Error in formula file:\nToo many clauses\n");
	fclose(stderr);
	exit(-1);
      }
      clause[cl].type = lit;
      clause[cl].lits = lit;
      if (maxlit < lit)
	maxlit = lit;
      lit = 0;
      cl++;
    } else {
      var = abs(num);
      if (var > N) {
	fprintf(stderr, "Error in formula file:\nToo many variables\n");
	fclose(stderr);
	exit(-1);
      }
      v[var].clauses++;
      if (v[var].clauses > maxconn)
	maxconn = v[var].clauses;
      lit++;
      literals++;
    }
  }
  allliterals = calloc(literals, sizeof(struct literalstruct));
  allclauses = calloc(literals, sizeof(struct clauselist));
  if(!allliterals || !allclauses) {
    fprintf(stderr, "Error:\nNot enough memory!\n");
  }
  for(var = 1; var <= N; var++) if (v[var].clauses) {
    v[var].clauselist = allclauses;
    allclauses += v[var].clauses;
    v[var].clauses = 0;
  }
  for(cl = 0; cl < M; cl++) {
    clause[cl].literal = allliterals;
    allliterals += clause[cl].lits;
  }
  //second pass to do the actual reading
  fprintf(stderr,". ");
  
  fclose(source);
  source = fopen(filename,"r");
  while ((aux = getc(source)) == 'c') {
    while (getc(source) != '\n');
  }
  ungetc(aux, source);
  fscanf(source, "p cnf %i %i", &N, &M);
  lit = 0; 
  cl = 0;
  while (fscanf(source, "%i ", &num) == 1) {
    if(!num) {
      lit = 0;
      cl++;
    } else {
      var = abs(num);
      v[var].clauselist[v[var].clauses].clause = clause + cl;
      v[var].clauselist[v[var].clauses++].lit = lit;
      clause[cl].literal[lit].var = var;
      clause[cl].literal[lit].bar = (num < 0);
      lit++;
    }
  }
   
  unused = 0;
  for (var = 1; var <= N; var++)
    if (v[var].clauselist == NULL) {
      v[var].spin = 3;
      unused++;
      fprintf(stderr, "WARNING!\n Var %i is unused!\n", var);
      
    }
  
  fclose(source);
  fprintf(stderr, "done\nformula read: %i cl, %i vars, %i unused\n%i literals, maxconn = %i,"
	  " maxliteral = %i c/v = %f\n", M, N, unused, literals, maxconn, maxlit, ((double) M) / N);
  
  return 0;
}


void randomformula(int K)
     //random k-sat formula
{
  int used,k,i,j,var,totcl=0;
  struct literalstruct *allliterals;
  struct clauselist *allclauses;
  
  clause=calloc(M,sizeof(struct clausestruct));
  v=calloc(N+1,sizeof(struct vstruct));
  allliterals=calloc(K*M,sizeof(struct literalstruct));
  allclauses=calloc(K*M,sizeof(struct clauselist));
  fprintf(stderr, "generating random formula with n=%i m=%i k=%i.",N,M,K);
  
  for(i=0; i<M; i++) {
    clause[i].type=K;
    clause[i].lits=K;
    clause[i].literal=allliterals;
    allliterals+=K;
    for(j=0; j<K; j++) {
      do {
	var=randint(N)+1;
	used=0;
	for(k=0;k<j;k++) {
	  if(var==clause[i].literal[k].var) {
	    used=1;
	    break;
	  }
	}
      } while (used);
      clause[i].literal[j].var=var;
      clause[i].literal[j].bar=randint(2);
      v[var].clauses++;
      totcl++;
      if(v[var].clauses>maxconn)
	maxconn=v[var].clauses;
    }
  }
  fprintf(stderr, ".");
  
  for(var=1; var<=N; var++) if(v[var].clauses){
    v[var].clauselist=allclauses;
    allclauses+=v[var].clauses;
    v[var].clauses=0;
  }
  fprintf(stderr, ".");
  
  for(i=0; i<M; i++) {
    for(j=0; j<K; j++) {
      var=clause[i].literal[j].var;
      v[var].clauselist[v[var].clauses].clause=clause+i;
      v[var].clauselist[v[var].clauses++].lit=j;
    }
  }
  maxlit=K;
  fprintf(stderr, " done\n");
  
}


int writeformula(FILE *sink)
     //write formula in dimacs cnf format
{
  int cl, lit, var, bar;
  int ncl = 0;
  
  for (cl = 0; cl < M; cl++) if (clause[cl].type == 0)
    ncl++;
  fprintf(sink, "p cnf %i %i\n", N, M - ncl);
  
  for (cl = 0; cl < M; cl++) if (clause[cl].type) {
    for (lit = 0; lit < clause[cl].lits; lit++) {
      var = clause[cl].literal[lit].var;
      bar = clause[cl].literal[lit].bar ? -1 : 1;
      if (v[var].spin == 0)
	fprintf(sink, "%i ", var * bar);
    }
    fprintf(sink, "0\n"); //outputs horrible just-zero lines
  }
  fflush(sink);
  return 0;
}
