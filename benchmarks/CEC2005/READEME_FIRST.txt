Readme file for C version of test problems.

Following is a brief description of included files.

1. global.h - contains global variable and function declarations.
Uncomment the (# define f*) line corresponding to a particular function, before attempting to compile the program.

2. sub.h - contains test problem specific declaration of global variables.

3. rand.h - contains random number generator related variables and routines.
SPRNG is being used as random number generator here.

4. aux.c - contains definitions of some auxillary routines.

5. def1.c - contains definitions of basic funcions (basis of hybrid composite and other functions).

6. def2.c - contains memory allocation and other routines.

7. def3.c - contains initialization routines.

8. def4.c - contains actual function definitions.

9. rand.c - contains random number generator related routines.

10. main.c - This is a sample file included that demonstrates, how to use various routines.
Go through this file carefully. Replace and/or edit this file to link the rest of the code with you main program.

Follow the following steps in order to use the code.

Step 1: Edit the file global.h - uncomment the line corresponding to the test function you want to use and comment all others.
Step 2: Edit the file main.c to suit your need (you may rename/replace it, if required). Remember to call initialization and memory allocation routines etc.
Step 3: compile the program with the Makefile provided. By default, it tries to compile all the source files present in the directory and link them and thefore avoid having redundant *.c files or edit the makefile to suit your need.
Note: File global.h contains global variables. Avoid using same names for your global variables if any.

Feel free to contact me in case of any questions/comments/suggestions/doubts.
__
Santosh Tiwari (tiwaris@iitk.ac.in)
Date: March 10, 2005
