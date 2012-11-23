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
#include <malloc.h>
#include "random.h"
#include "formula.h"
#include "spy.h"

extern struct clausestruct *clause;
extern struct vstruct *v;
extern int M;
extern int N;
extern int freespin;
extern int verbose;
extern FILE *replay;
extern int maxconn;
extern int maxlit;
extern struct literalstruct *near;
extern int *vic;
extern FILE *out_file;


int fix(int var, int spin)
//fix var to value spin and possibly simplify the resulting formula
{
  if (spin)      
    freespin--;
  else
    freespin++;
  
  if (v[var].spin == spin)
    return 1;
  
  v[var].spin = spin;
  return 0;
}


int writesolution(FILE *sink)
     //write the solution found so far to a file
{
  int var;
  
  for (var = 1; var <= N; var++){
    if (v[var].spin) {
      fprintf(sink, "%i\n", (v[var].spin > 0) ? var : -var);
    }		  
  }
  fflush(sink);
  return 0;
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
    fprintf(out_file, "Error in formula file!\n Unable to read %s.\n", filename);
    fclose(out_file);
    exit(-1);
  }
  fprintf(out_file, "reading variable clause-size formula %s ", filename);
  //skip comments
  while ((aux = getc(source)) == 'c') {
    while(getc(source) != '\n');
  }
  ungetc(aux, source);
  fprintf(out_file, ".");
  fflush(out_file);
  //read p line
  fscanf(source, "p cnf %i %i", &N, &M);
  v = calloc(N + 1, sizeof(struct vstruct));
  clause = calloc(M, sizeof(struct clausestruct));
  //first pass for counting
  fprintf(out_file, ".");
  fflush(out_file);
  while (fscanf(source, "%i ", &num) == 1) {
    if (!num) {
      if (cl == M) {
	fprintf(out_file, "Error in formula file:\nToo many clauses\n");
	fclose(out_file);
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
	fprintf(out_file, "Error in formula file:\nToo many variables\n");
	fclose(out_file);
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
    fprintf(out_file, "Error:\nNot enough memory!\n");
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
  fprintf(out_file,". ");
  fflush(out_file);
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
  freespin = N;
  
  unused = 0;
  for (var = 1; var <= N; var++)
    if (v[var].clauselist == NULL) {
      v[var].spin = 3;
      unused++;
      fprintf(out_file, "WARNING!\n Var %i is unused!\n", var);
      fflush(out_file);
    }
  
  fclose(source);
  fprintf(out_file, "done\nformula read: %i cl, %i vars, %i unused\n%i literals, maxconn = %i,"
	  " maxliteral = %i c/v = %f\n", M, N, unused, literals, maxconn, maxlit, ((double) M) / N);
  fflush(out_file);
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
  fprintf(out_file, "generating random formula with n=%i m=%i k=%i.",N,M,K);
  fflush(out_file);
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
  fprintf(out_file, ".");
  fflush(out_file);
  for(var=1; var<=N; var++) if(v[var].clauses){
    v[var].clauselist=allclauses;
    allclauses+=v[var].clauses;
    v[var].clauses=0;
  }
  fprintf(out_file, ".");
  fflush(out_file);
  for(i=0; i<M; i++) {
    for(j=0; j<K; j++) {
      var=clause[i].literal[j].var;
      v[var].clauselist[v[var].clauses].clause=clause+i;
      v[var].clauselist[v[var].clauses++].lit=j;
    }
  }
  maxlit=K;
  freespin=N;
  fprintf(out_file, " done\n");
  fflush(out_file);
}


void close_computation(void)
     //simplifies formula according to the list of fixed spins
{
  fprintf(out_file, "\nFormula simplified!\n\n");
  fflush(out_file);
  fclose(out_file);
  system("./verify " SOLUTION " " FORMULA " " SUBFORMULA " >> " OUT_FILE);
  exit(0);
}
























