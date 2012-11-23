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

#include <stdio.h>

//buffer for executing extern shell commands
#define BUFFERSIZE 1000


struct weightstruct {
  double p;
  double z;
  double m;
};


struct literalstruct {
  int var;   //varnum=1,...,N
  unsigned char bar;  //bar=0,1
  double eta; //eta of the literal Q(u)=delta(u)+eta*delta(u+bar?-1:1)
  struct weightstruct H;  //contains cavity integrated H-survey
};

struct clausestruct {
  struct literalstruct *literal; //list of literals
  int type;     //type=0,1,... actual number of lits
  int lits;     //lits=0,1,... initial number of lits
};

struct clauselist {
  struct clausestruct *clause; //in which clause
  unsigned char lit; //in which literal on such clause
};

struct vstruct {
  unsigned char clauses;           //in how many clauses the var appears
  struct clauselist *clauselist;   //list of clauses
  int spin;                        //spin of the var, 0=unfixed
  struct weightstruct H;           //contains local integrated H-survey
};

void initformula();
int readformula(FILE * source);
int readvarformula(char *);
void randomformula();
int writeformula(FILE *sink);
int writesolution(FILE *sink);
int simplify(int var);
int fix(int var, int spin);
void print_stats();
void close_computation(void);



