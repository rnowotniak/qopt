/*
    Copyright 2004 Demian Battaglia, Alfredo Braunstein, Michal Kolar,
    Michele Leone, Marc Mézard, Martin Weigt and Riccardo Zecchina

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
#include <unistd.h>
#include "random.h"
#include "formula.h"
#include "spy.h"
#include "bk.h"


//extern variables 
extern double y;               //pseudotemperature
extern struct vstruct *v;      //all per var information
extern struct clausestruct *clause; //all per clause information
extern int *perm;              //permutation, for update ordering
extern int M;                  //number of clauses
extern int N;                  //number of variables
extern int *ncl;               //clause size histogram
extern int maxconn;            //maximal connectivity
extern double epsilon;         //convergence criterion
extern int iterations;         //maximal number of iterations
extern FILE *out_file;         //output file
extern double *hsurvey_store, *oldhsurvey_store;
extern int verbose;

//global variables
double expymm;                 //exponential exp(-2y)


void hsurvey_stores(int maxconn)
     //allocates memory for the compute_field_bk
{
  hsurvey_store = (double *) calloc(2 * maxconn + 3, sizeof(double));
  oldhsurvey_store = (double *) calloc(2 * maxconn + 3, sizeof(double));
  test_array(hsurvey_store);
  test_array(oldhsurvey_store);
  return;
}

int converge(int write_flag)
     //call iterate() until the maximum difference of eta with previous 
     //step is small.
{
  double eps = 0;
  int iter = 0, cl, m;
  
  for(cl = 0, m = 0; cl < M; cl++) if (clause[cl].type) {
    perm[m++] = cl;
  }
  
  do {
    eps = iterate();
  } while (eps > epsilon && iter++ < iterations);
  
  if (eps <= epsilon) {
    if (verbose >= 2){
      fprintf(out_file,"[%f] :o)\n", eps);
      fflush(out_file);
    }
    return 1;
  } else {
    fprintf(out_file,"[%f] :o(\n", eps);
    fflush(out_file);
    if (write_flag) {
       writesolution(fopen(SOLUTION, "w+"));
       close_computation();
    }
    return 0;
  }
}


struct weightstruct compute_field(int var, int local_flag)
     //computes H-survey for the variable var node. It computes the local field 
     //of the variable or the cavity field of a literal depending on having set 
     //to zero the type of a clause before calling it or not
{
  struct clauselist *cl;
  struct weightstruct eta, H;
  struct literalstruct literal;
  int conn, type, s, m, h, t, flag = 0;
  double *hsurvey, *oldhsurvey, *auxhsurvey, norm;

  if (var < 0) {
    var = -var;
    flag = 1;
  }
  
  if (!local_flag)            //when computing cavity fields for fixed variable
    if ((s = v[var].spin)) {  //send always fully polarized bias. This has not
      H.p = s > 0 ? 0.0 : 1.0;//to be done when computing local fields
      H.z = 0.0; 
      H.m = 1.0 - H.p;
      return H;
    }
  
  //clean the workspace
  for (m = 0; m < 2 * maxconn + 3; m++) {
    oldhsurvey_store[m] = 0;
    hsurvey_store[m] = 0;
  } 
  
  cl = v[var].clauselist;
  conn = v[var].clauses;

  for (m = 0; m < conn; m++) {  //find the first nontrivial clause
    if ((type = cl[m].clause->type)) break;
  }
  if ((m == conn) && (cl[m - 1].clause->type == 0)) {  
    H.m = 0.0; H.z = 1.0; H.p = 0.0;  //return paramagnetic H-survey if all
    return H;                         //clauses are trivial
  }
  
  if (m == conn) m--;

  hsurvey = hsurvey_store + conn + 1;   //shift pointers to make code readable
  oldhsurvey = oldhsurvey_store + conn + 1;

  //load parameters of first untrivial clause and initialize  
  literal = cl[m].clause->literal[cl[m].lit];
  hsurvey[ (literal.bar) ? -1 : 1 ] = literal.eta;
  hsurvey[0]  = 1 - literal.eta;
  
/*  ------------------------------------------------------------------- */
/*     ITERATIVE COMPUTATION OF CAVITY OR LOCAL H-SURVEY                 */
/*  ------------------------------------------------------------------- */

  for (t = 1; t < conn - m; t++) {                
    if (cl[m + t].clause->type == 0) continue;
    
    auxhsurvey = oldhsurvey;
    oldhsurvey = hsurvey;
    hsurvey    = auxhsurvey;
    
    literal = cl[m + t].clause->literal[cl[m + t].lit];
  
    eta.z = 1 - literal.eta;
    eta.p = literal.eta;    //only one of them is used, only one is nonzero
    eta.m = literal.eta;
    
    if (literal.bar) {
      for (h = -t - 1; h < 0; h++) //negative fields 
	hsurvey[h] = eta.z * oldhsurvey[h] + eta.m * oldhsurvey[h + 1];
      for (h = 0; h <= t + 1; h++) //zero and positive fields 
	hsurvey[h] = eta.z * oldhsurvey[h] + eta.m * expymm * oldhsurvey[h + 1];
    } else {
      for (h = -t - 1; h <= 0; h++)//negative and zero fields 
	hsurvey[h] = eta.z * oldhsurvey[h] + eta.p * expymm * oldhsurvey[h - 1];
      for (h = 1; h <= t + 1; h++) //positive fields 
	hsurvey[h] = eta.z * oldhsurvey[h] + eta.p * oldhsurvey[h - 1];
    }
  }
  
/*  ------------------------------------------------------------------- */
/*     END OF SURVEY COMPUTATION                                         */
/*  ------------------------------------------------------------------- */

  H.p = 0;                     //Compute integrals of h-survey (weights)
  H.z = hsurvey[0];
  H.m = 0;

  //outputs probability of being pozitive
  for (h = 1; h <= conn; h++) {//or negative
    H.m += hsurvey[h];
    H.p += hsurvey[-h];
  }
  
  if (flag) {
    norm = H.m + H.z + H.p;
    H.m = 0; H.z = 0; H.p = 0;
    for (h = 1; h <= conn; h++) {//outputs mean value
      H.m += h * hsurvey[h];
      H.p += h * hsurvey[-h];
    }
    H.m /= norm;
    H.p /= norm;
  }

  return H;
}


void expy(double y)
     //computes exp(-y) and exp(-2y)
{
  if (y > 20)
    expymm = 0.0; 
  else
    expymm = exp(-2.0 * y);
}


double iterate(void)
     //update etas of clauses in a random permutation order
{
  int cl, m, i;
  double eps, maxeps;
  
  eps = 0.0;
  maxeps = 0.0;
  
  for (m = M; m; m--) {
    cl = perm[i = randint(m)];
    perm[i] = perm[m - 1];
    perm[m - 1] = cl;
  
    eps = update_eta(cl);
    
    if (maxeps < eps)
      maxeps = eps;
  }
  if (verbose >=2) {
    fprintf(out_file, ".");
    fflush(out_file);
  }
  return maxeps;
}


double update_eta(int cl)
     //updates all eta's for a given clause cl
{
  int l, k;
  int new_type, n_lit;
  struct clausestruct *klaus;
  struct literalstruct literal, kiteral;
  double eps, maxeps, neweta, norm;
  struct weightstruct H;
  
  klaus = &(clause[cl]);
  maxeps = 0.0;
  eps = 0.0;
  
  //Set type to zero in order to compute cavity fields and not local fields.    
  if (klaus->type == 1) {    //if one-clause keep old eta and return 
    return 0.0;              //maxeps = 0
  }

  n_lit = klaus->lits;
  new_type = 0;
  klaus->type = 0;
  
  //Compute all cavity fields.
  for (l = 0; l < n_lit; l++) { // if (v[klaus->literal[l].var].spin == 0) {
    H = compute_field(klaus->literal[l].var, 0);
    norm = (H.m + H.z + H.p);
    H.m /= norm; H.z /= norm; H.p /= norm;
    klaus->literal[l].H = H;
    new_type++;
  }
  
  //Loop over all the literals: Fix one literal and compute its eta.  
  for (l = 0; l < n_lit; l++) { //if (v[klaus->literal[l].var].spin == 0) {
    literal = klaus->literal[l];
    neweta = 1;
    
    for (k = 0; k < klaus->lits; k++) { //if (v[klaus->literal[k].var].spin == 0) {
      if (k == l) continue;           //chooses the other literals of the clause
      
      kiteral = klaus->literal[k];
      neweta *= (kiteral.bar) ? kiteral.H.m : kiteral.H.p;     
    }
    
    //  maxeps is the largest eps:             
    eps = fabs(neweta - clause[cl].literal[l].eta);
    if (maxeps < eps)
      maxeps = eps;                                                           
     
    //Store the new eta
    clause[cl].literal[l].eta = (neweta > EPS) ? neweta : 0;
  }
  
  klaus->type = new_type;      //update the type of the clause
  return maxeps;
}


double compute_free_energy(void)
     //computation of the zero temperature free energy (Phi(y))
{
  double phifun, phivar, wprod, cprod;
  int m, n, neigh, newtype;
  struct clausestruct *cl;
  struct clauselist *cli;
  struct weightstruct H;

  phifun = 0.0;
  phivar = 0.0;

  for (m = 0; m < M; m++) if (clause[m].type) { //contribution of clauses
    cl = &(clause[m]);                          //(FUNCTION NODES)
    cl->type = 0;                //set type to 0 to compute cavity fields
    newtype = 0;
    wprod = 1.0;
    cprod = 1.0;
    for (n = 0; n < cl->lits; n++) { //if (v[cl->literal[n].var].spin == 0) {
      newtype++;
      H = compute_field(cl->literal[n].var, 0);  //update cavity fields
      wprod *= (cl->literal[n].bar ? H.m : H.p);
      cprod *= (H.m + H.z + H.p);
    }
    phifun += -log(cprod + (expymm - 1.0) * wprod) / y;
    cl->type = newtype;                        //restore the type of the clause
  }

  for (n = 1; n <= N; n++) if (v[n].spin == 0) {//contribution of var nodes

    neigh = 0;
    cli = v[n].clauselist;
    for (m = 0; m < v[n].clauses; cli++, m++)
      if (cli->clause->type)
        neigh++;                                //count the nonfixed clauses
                                                //neighbouring to var[n]
    if (neigh < 2) continue;
    H = compute_field(n, 1);
    phivar += -(log(H.m + H.p + H.z)) * (((double) neigh) - 1.0) / y;
  }

  return  (phifun - phivar);
}


void graph_free_energy()
     //Produces free energy plot against y and returns optimal y
{
  double y_opt, free_energy, free_old;
  static double max_y = MAX_Y;
  int decrease;
  FILE *fef;

  y_opt = y;
  free_old = -1.0e300;   //large negative number
  free_energy = -1.0e300;

  if ((fef = fopen(FREE_ENERGY_FILE, "a")) == NULL) {
    fprintf(out_file, "Cannot open file for storing the free energy data\n");
    fclose(out_file);
    exit(-1);
  }
  if (verbose) {
    fprintf(out_file, "Finding by bisection the maximum of free energy...\n");
    fflush(out_file);
  }
  rtbis(MIN_Y, max_y);       //finds the last value of y for which 
                             //the system converges
  decrease = 0;              //counter of succesive decreases

  fprintf(fef, "\n*\n");
  for (; 1; y -= STEP_Y) {
    expy(y);

    if (converge(0)) {
      free_old = free_energy;
      free_energy = compute_free_energy() / ((double) 2); 
                                            //Divide by two because of energy 
                                            //scale conventions
      if ((free_energy > free_old) && (fabs(free_energy) > EPS)) {
        y_opt = y;
	decrease = 0; //zero the counter of consecutive decreases
      } else {      
	decrease++; //increase the counter of consecutive decreases  
      }
      fprintf(fef, "y / free energy / change = %1.3e, %1.3e, %1.3e\n",
              y, free_energy, free_energy - free_old);
      if (decrease == 5)
	break;        //test the counter for the desired quota
    } else {
      randomize_eta();  //randomize eta's
      y -= STEP_Y;      //jump faster
    }
    if ((fabs(free_energy) < EPS) || (fabs(free_energy - free_old) > PHASE_TRANS))
      randomize_eta();                         //if system falls 
  }                                            //in paramagnetic state
                                               //randomize eta's  
  y = y_opt;
  max_y = y + 5 * STEP_Y;

  fclose(fef);

  fprintf(out_file, "------------------------------\n");
  fprintf(out_file, "Optimal y found: y = %1.3e\n", y);
  fprintf(out_file, "------------------------------\n");
  fflush(out_file);
}


void rtbis(double y1, double y2)
     //bisection to find the largest y where the procedure 
     //converges, to be called with y2 = MAX_Y and y1 = MIN_Y 
{ 
  int j, f, fmid; 
  double dy, rtb;

  iterations = UPDATE_ITERATIONS;
    
  y = y2;
  f = converge_with_(y);

  if (f == 1) {
    iterations = ITERATIONS;
    return;
  }

  y = y1;
  fmid = converge_with_(y);

  if ((f + fmid) == 0) {
    fprintf(out_file, "We encountered serious problems when converging,\n"
	              "returning current best solution.\n"); 
    writesolution(fopen(SOLUTION, "w+"));
    close_computation();
    exit(-1);
  }

  dy = y2 - y1; 
  rtb = y1;

  for (j = 1; j <= 11; j++) { 
    y = rtb + (dy *= 0.5);
    fmid = converge_with_(y);
    if (fmid == 1)
      rtb = y;
    if (fabs(dy) < ACCURACY_Y) break;
  } 
  y -= dy;
  iterations = ITERATIONS;
  if (verbose){
    fprintf(out_file, "\nFine-tuning:\n");
    fflush(out_file);
  }
  return; 
}


void backup(void) 
     //Do backup before computing free energy
{
  FILE *bf, *bs;
  static int n = 0;
  
  if (n) { 
    system("mv backup.tmp.cnf backup.old.cnf");
    system("mv backup.tmp.sol backup.old.sol");
  }
  n = 1;  
 
  if ((bf = fopen("backup.tmp.cnf", "w")) == NULL) {
    fprintf(out_file, "Cannot open backup file!\n");
    fclose(out_file);
    exit(-1);
  }
  if ((bs = fopen("backup.tmp.sol", "w")) == NULL) {
    fprintf(out_file, "Cannot open backup file!\n");
    fclose(out_file);
    exit(-1);
  }
  
  writeformula(bf);
  writesolution(bs);

  if (fclose(bf) == EOF) {
    fprintf(out_file, "Cannot close backup file.\n");
    fclose(out_file);
    exit(-1);
  }
  if (fclose(bs) == EOF) {
    fprintf(out_file, "Cannot close backup file.\n");
    fclose(out_file);
    exit(-1);
  }
}













