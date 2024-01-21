include ./arch.gnu
# OPTIMIZATION = -fast
# OPTIMIZATION = -O3 -fopenmp
# DEBUG += -g

all: cardiacsim_serial cardiacsim_parallel_1 cardiacsim_parallel_2 cardiacsim_parallel_3

# app:		cardiacsim_serial

SHARED_OBJECTS = splot.o cmdLine.o
SERIAL_OBJECTS = cardiacsim_serial.o $(SHARED_OBJECTS)
PARALLEL_1_OBJECTS = cardiacsim_parallel_1.o $(SHARED_OBJECTS)
PARALLEL_2_OBJECTS = cardiacsim_parallel_2.o $(SHARED_OBJECTS)
PARALLEL_3_OBJECTS = cardiacsim_parallel_3.o $(SHARED_OBJECTS)


cardiacsim_parallel_3:	        $(PARALLEL_3_OBJECTS) 
	$(C++LINK) $(LDFLAGS) -o $@ $(PARALLEL_3_OBJECTS)  $(LDLIBS)

cardiacsim_parallel_2:	        $(PARALLEL_2_OBJECTS) 
	$(C++LINK) $(LDFLAGS) -o $@ $(PARALLEL_2_OBJECTS)  $(LDLIBS)

cardiacsim_parallel_1:	        $(PARALLEL_1_OBJECTS) 
	$(C++LINK) $(LDFLAGS) -o $@ $(PARALLEL_1_OBJECTS)  $(LDLIBS)

cardiacsim_serial:	        $(SERIAL_OBJECTS) 
	$(C++LINK) $(LDFLAGS) -o $@ $(SERIAL_OBJECTS)  $(LDLIBS)

clean:	
	$(RM) *.o cardiacsim_serial cardiacsim_parallel_1 cardiacsim_parallel_2 cardiacsim_parallel_3 *~;
	$(RM) core;
