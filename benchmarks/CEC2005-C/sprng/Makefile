############################################################################
#
# Then typing the command below   => results in the following being created
#      make               => SPRNG libraries and example programs
#      make src		  => SPRNG libraries (in ./lib) and certain executables
#      make examples	  => SPRNG examples
#      make tests	  => Tests of quality of random streams
#
# Object files created during the compilation process can be deleted finally
# by typing
#       make clean
#
# Object files, executables, and the libraries can be deleted by typing
#       make realclean
############################################################################

SHELL = /bin/sh

include make.CHOICES

LIBDIR = $(LIB_REL_DIR)
SRCDIR = SRC
DIRS = SRC EXAMPLES TESTS lib

include $(SRCDIR)/make.$(PLAT)

all : src examples tests

#---------------------------------------------------------------------------
src :
	(cd SRC; $(MAKE) LIBDIR=../$(LIBDIR) SRCDIR=../$(SRCDIR) PLAT=$(PLAT); cd ..)

examples : 
	(cd EXAMPLES; $(MAKE) LIBDIR=../$(LIBDIR) SRCDIR=../$(SRCDIR) PLAT=$(PLAT))

tests : 
	(cd TESTS; $(MAKE) LIBDIR=../$(LIBDIR) SRCDIR=../$(SRCDIR) PLAT=$(PLAT))

#---------------------------------------------------------------------------
clean :
	@for l in $(DIRS) ; do \
	  cd $$l ; \
	  $(MAKE) PLAT=$(PLAT) clean ; \
	  cd .. ; \
        done

realclean :
	@for l in $(DIRS) ; do \
	  cd $$l ; \
	  $(MAKE) PLAT=$(PLAT) realclean ; \
	  cd .. ; \
        done
	@rm -f core *~ check* time* *.data

.SUFFIXES : 
