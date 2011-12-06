############################################################################
# This is a sample make file to compile some example programs
#
# We have set variables giving information on the archiver, the C compiler 
# and the FORTRAN compiler for certain machines in files make.xxx, where xxx 
# is the machine name, set using the PLAT variable in ../make.CHOICES. 
#
# Then typing the command below    => results in the following being created
#      make                        => Same as: make serial.
#      make serial                 => Single processor examples.
#
#      make mpi                    => Examples using MPI
#
#      make all                    => Single processor and MPI examples.
#
#
# Object files created during the compilation process can be deleted 
# by typing
#       make clean
#
# Set 'LIBDIR' to the directory where the SPRNG libraries are. 
# Set 'INCDIR' to the directory where the SPRNG include files are. 
# We have already set these correctly, unless whoever installed the libraries
# changed the locations.
############################################################################

SHELL = /bin/sh

include ../make.CHOICES

LIBDIR  = ../$(LIB_REL_DIR)
SRCDIR = ../SRC
INCDIR = ../include

# use 'lfg' to get Lagged Fibonacci, 'lcg' to get Linear Congruential, etc. 
SPRNGLIB=lcg
##########################################################################

EX = sprng seed checkpoint invalid_ID sprng-simple simple-simple seed-simple \
     checkpoint-simple spawn sprngf seedf checkpointf invalid_IDf \
     spawnf sprngf-simple simplef-simple seedf-simple checkpointf-simple \
     pi-simple  pif-simple

MPIEX = sprng_mpi fsprng_mpi 2streams_mpi seed_mpi message_mpi \
        sprng-simple_mpi fsprng-simple_mpi seed-simple_mpi message-simple_mpi \
        sprngf_mpi fsprngf_mpi seedf_mpi messagef_mpi 2streamsf_mpi  \
        sprngf-simple_mpi fsprngf-simple_mpi seedf-simple_mpi \
        messagef-simple_mpi pi-simple_mpi

include $(SRCDIR)/make.$(PLAT)

serial : $(EX)

all: $(EX) $(MPIEX)

mpi : $(MPIEX)

simple-simple : simple-simple.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) -I$(INCDIR) -o simple-simple simple-simple.c -L$(LIBDIR) -l$(SPRNGLIB)

sprng : sprng.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) $(CHK) -I$(INCDIR) -o sprng sprng.c -L$(LIBDIR) -l$(SPRNGLIB)

sprng-simple : sprng-simple.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) -I$(INCDIR) -o sprng-simple sprng-simple.c -L$(LIBDIR) -l$(SPRNGLIB)

sprng-simple_mpi : sprng-simple_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) -I$(INCDIR) -o sprng-simple_mpi sprng-simple_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

sprng_mpi : sprng_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) $(CHK) -I$(INCDIR) -o sprng_mpi sprng_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

fsprng_mpi : fsprng_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) $(CHK) -I$(INCDIR) -o fsprng_mpi fsprng_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

fsprng-simple_mpi : fsprng-simple_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) -I$(INCDIR) -o fsprng-simple_mpi fsprng-simple_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

2streams_mpi : 2streams_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) $(CHK) -I$(INCDIR) -o 2streams_mpi 2streams_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

seed : seed.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) $(CHK) -I$(INCDIR) -o seed seed.c -L$(LIBDIR) -l$(SPRNGLIB)

seed-simple : seed-simple.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) -I$(INCDIR) -o seed-simple seed-simple.c -L$(LIBDIR) -l$(SPRNGLIB)

seed_mpi : seed_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) $(CHK) -I$(INCDIR) -o seed_mpi seed_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

seed-simple_mpi : seed-simple_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) -I$(INCDIR) -o seed-simple_mpi seed-simple_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

checkpoint : checkpoint.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) $(CHK) -I$(INCDIR) -o checkpoint checkpoint.c -L$(LIBDIR) -l$(SPRNGLIB)

checkpoint-simple : checkpoint-simple.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) -I$(INCDIR) -o checkpoint-simple checkpoint-simple.c -L$(LIBDIR) -l$(SPRNGLIB)

message_mpi : message_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) $(CHK) -I$(INCDIR) -o message_mpi message_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

message-simple_mpi : message-simple_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) -I$(INCDIR) -o message-simple_mpi message-simple_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

pi-simple : pi-simple.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) $(CHK) -I$(INCDIR) -o pi-simple pi-simple.c -L$(LIBDIR) -l$(SPRNGLIB) -lm

pi-simple_mpi : pi-simple_mpi.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPICC) $(CFLAGS) -I$(INCDIR) -o pi-simple_mpi pi-simple_mpi.c -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB) -lm

spawn : spawn.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) -DCHECK_POINTERS -I$(INCDIR) -o spawn spawn.c -L$(LIBDIR) -l$(SPRNGLIB)

invalid_ID : invalid_ID.c $(LIBDIR)/lib$(SPRNGLIB).a
	$(CC) $(CFLAGS) -DCHECK_POINTERS -I$(INCDIR) -o invalid_ID invalid_ID.c -L$(LIBDIR) -l$(SPRNGLIB)

sprngf : sprngf.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o sprngf  sprngf.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB)

sprngf_mpi : sprngf_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o sprngf_mpi  sprngf_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

fsprngf_mpi : fsprngf_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o fsprngf_mpi  fsprngf_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

seedf : seedf.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o seedf  seedf.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB)

seedf_mpi : seedf_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o seedf_mpi  seedf_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

checkpointf : checkpointf.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o checkpointf  checkpointf.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(CMDDIR) $(CMDLIB)

messagef_mpi : messagef_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o messagef_mpi  messagef_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)


2streamsf_mpi : 2streamsf_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o 2streamsf_mpi  2streamsf_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

invalid_IDf : invalid_IDf.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o invalid_IDf  invalid_IDf.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB)

spawnf : spawnf.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o spawnf  spawnf.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB)


simplef-simple : simplef-simple.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o simplef-simple  simplef-simple.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB)

sprngf-simple : sprngf-simple.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o sprngf-simple  sprngf-simple.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB)

sprngf-simple_mpi : sprngf-simple_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o sprngf-simple_mpi  sprngf-simple_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

fsprngf-simple_mpi : fsprngf-simple_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o fsprngf-simple_mpi  fsprngf-simple_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

seedf-simple : seedf-simple.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o seedf-simple  seedf-simple.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB)

seedf-simple_mpi : seedf-simple_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o seedf-simple_mpi  seedf-simple_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

checkpointf-simple : checkpointf-simple.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o checkpointf-simple  checkpointf-simple.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(CMDDIR) $(CMDLIB)

messagef-simple_mpi : messagef-simple_mpi.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(MPIF77) $(FFLAGS)  -I$(INCDIR) -o messagef-simple_mpi  messagef-simple_mpi.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) $(MPIDIR) $(MPILIB)

pif-simple : pif-simple.$(FSUFFIX) $(LIBDIR)/lib$(SPRNGLIB).a
	$(F77) $(FFLAGS)  -I$(INCDIR) -o pif-simple  pif-simple.$(FSUFFIX) -L$(LIBDIR) -l$(SPRNGLIB) -lm


.SUFFIXES : 

.SUFFIXES : .f .F

.F.f : 
	@if [ -f $*.i ] ; then \
	rm $*.i ;\
	fi 
	$(CPP) -I$(INCDIR) $*.F
	@if [ -f $*.i ] ; then \
	mv $*.i  $*.f ;\
	fi 

clean :
	rm -f *.o

realclean :
	rm -f *.o $(EX) $(MPIEX) *~ *.i core a.out
