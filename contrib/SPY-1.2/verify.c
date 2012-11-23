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

//header lines
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
//macros
#define test_array(array) if((array)==NULL){fprintf(stderr,"Error: Not enough memory for internal structures.");exit(-1);}
//function prototypes
void read_formula(void);
void read_solution(void);
void decimate(void);
void write_subformula(void);
void output_unsat(void);
//end of header

FILE *ff, *fs, *fo;
int M, N;
int fixed, sat, unsat;
int K = 3;          //K-SAT formula
int **formula;
int **decimated;
int *solution;
int *satisfied;     //0 if UNSAT, 1 if SAT


int main(int argc, char ** argv)
     //opens all files and ...
{
  int m;

  if ((ff = fopen(argv[2],"r")) == NULL) {
    printf("Cannot open the input formula file, the second argument of the command line.\n");
    exit(-1);
  }
  if ((fs = fopen(argv[1],"r")) == NULL) {
    printf("Cannot open the solution file, the first argument of the command line.\n");
    exit(-1);
  }
  if ((fo = fopen(argv[3],"w")) == NULL) {
    printf("Cannot open the output formula file, the third argument of the command line.\n");
    exit(-1);
  }

  read_formula();
  read_solution();
  decimate();
  write_subformula();
  printf("Subformula stored in output file.\n");
  if (unsat) output_unsat();

  for (m = 1; m <= M; m++) {
    cfree(decimated[m]);
    cfree(formula[m]);
  }
  cfree(formula), cfree(decimated), cfree(satisfied), cfree(solution);
  //output macchine readable data  
  printf("\n#DATA(N, fixed vars, M, SAT clauses, UNSAT clauses)\n%d\t%d\t%d\t%d\t%d\n", N, fixed, M, sat, unsat);
  return 1;
}


void read_formula(void)
     //reads the formula fila and stores the formula in the array formula
{
  int aux;
  int m, n;

  fscanf(ff, "p cnf %d %d\n", &N, &M);
  formula = calloc(M + 2, sizeof(int*));
  test_array(formula);
  
  for (m = 1; m <= M; m++) {
    formula[m] = calloc(K + 1, sizeof(int));
    test_array(formula[m]);
  }
  
  for (m = 1; m <= M; m++) {
    n = 1;
    fscanf(ff, "%d", &aux);
    if (aux) formula[m][n++] = aux;
    while (getc(ff) == ' ') {
      fscanf(ff, "%d", &aux);
      if (aux) formula[m][n++] = aux;
    }
  }
  printf("Read formula with %d clauses and %d variables.\n", M, N);
} 
 
  
void read_solution(void)
     //reads solution and stores it in the array solution
{
  int aux;
  
  solution = calloc(N + 2, sizeof(int));
  test_array(solution);

  fixed = 0;
  while (aux = getc(fs), !feof(fs)) { 
    ungetc(aux, fs);
    fscanf(fs, "%d\n", &aux);
    solution[abs(aux)] = aux;
    fixed++;
  }
  printf("Read solution with %d fixed variables.\n\n*\n", fixed);
}


void decimate(void)
     //decimates the formula with fixed spins, returns number
     //unfixed clauses
{
  int m, k = 1;

  unsat = 0;
  decimated = calloc(M + 2, sizeof(int*));
  test_array(decimated);

  for (m = 1; m <= M; m++) {
    decimated[m] = calloc(K + 1, sizeof(int));
    test_array(decimated[m]);
  }

  for (m = 1; m <= M; m++)
    for (k = 1; k <= K; k++)
      decimated[m][k] = formula[m][k];
 
  satisfied = calloc(M + 2, sizeof(int));
  test_array(satisfied);

  for (m = 1; m <= M; m++) {
    for (k = 1; k <= K; k++)
      if (formula[m][k]) 
	satisfied[m]++;
    if (!satisfied[m]) {
      printf("Clause %d empty, trivially satisfied.\n", m);
      satisfied[m] = -17;
    }  
  }
  
  for (m = 1; m <= M; m++) {
    for (k = 1; k <= K; k++) {
      if (decimated[m][k] == 0) break;
      else {
	if (decimated[m][k] == solution[abs(decimated[m][k])]) {
	  satisfied[m] = -17;
	  break;
	} 
	else if (solution[abs(decimated[m][k])]) {
	  decimated[m][k] = 0;
	  satisfied[m]--;
	  if (satisfied[m] == 0) { 
	    printf("Clause %d UNSAT.\n", m);
	    unsat++;
	  }
	}
      }
    }
  }
  
  sat = 0;
  for (m = 1; m <= M; m++) if (satisfied[m] == -17)
    sat++;
  printf("Alltogether %d clauses satisfied.\n", sat);
}


void write_subformula(void)
     //writes subformula
{
  int m, k, maxn = 0;
  int aux;

  for (m = 1; m <= M; m++) if (satisfied[m] > 0) {
    for (k = 1; k <= K; k++)
      if (decimated[m][k] > maxn)
	maxn = decimated[m][k];
  }
  
  fprintf(fo, "p cnf %d %d\n", maxn, M - sat - unsat);
  
  for (m = 1; m <= M; m++) if (satisfied[m] > 0) {
    aux = 0;
    for (k = 1; k <= K; k++)
      if (decimated[m][k]) {
	aux = 1;
	fprintf(fo, "%d ", decimated[m][k]);
      }
    if (aux) fprintf(fo, "0\n");
  }
}


void output_unsat(void)
     //writes UNSAT clauses to stdout
{
  int m, k, aux;
  
  printf("List of %d UNSAT clauses:\n", unsat);
  
  for (m = 1; m <= M; m++) if (satisfied[m] == 0) {
    aux = 0;
    for (k = 1; k <= K; k++)
      if (formula[m][k]) {
        aux = 1;
        printf("%d ", formula[m][k]);
      }
    if (aux) printf("0\n");
  }
}
