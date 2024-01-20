include ./arch.gnu
# OPTIMIZATION = -fast
# OPTIMIZATION = -O3
# DEBUG += -g

all: cardiacsim_serial cardiacsim_parallel_1

# app:		cardiacsim_serial

SHARED_OBJECTS = splot.o cmdLine.o
SERIAL_OBJECTS = cardiacsim_serial.o $(SHARED_OBJECTS)
PARALLEL_1_OBJECTS = cardiacsim_parallel_1.o $(SHARED_OBJECTS)


cardiacsim_parallel_1:	        $(PARALLEL_1_OBJECTS) 
	$(C++LINK) $(LDFLAGS) -o $@ $(PARALLEL_1_OBJECTS)  $(LDLIBS)

cardiacsim_serial:	        $(SERIAL_OBJECTS) 
	$(C++LINK) $(LDFLAGS) -o $@ $(SERIAL_OBJECTS)  $(LDLIBS)

clean:	
	$(RM) *.o cardiacsim_serial cardiacsim_parallel_1 *~;
	$(RM) core;
