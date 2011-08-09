CC = g++
#CC = /home/bamboos/usb/intel/bin/ia32/icc
CFLAGS = -O3 -Wall
CPPFLAGS = -O3 -Wall -D LINUX -D _HAS_ITERATOR_DEBUGGING=0 -D _SECURE_SCL=0 -D BOOST_DISABLE_ASSERTS
#CPPFLAGS = -O3 -Wall -D ICP_USE_THREADS
#CFLAGS = -O3 -Wall -D PROFILER_USE=0
OBJ = main.o Math.o
LIBS = -lboost_program_options

evaluator: $(OBJ)
	$(CC) $(CPPFLAGS) $^ -o $@ -lm $(LIBS)

all: evaluator

clean:
	rm -f $(OBJ) evaluator

.PHONY: all clean
