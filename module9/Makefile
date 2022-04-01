thrust_objects = thrust.o

all: thrust.exe

thrust.exe: $(thrust_objects)
	nvcc $(thrust_objects) -o $@

%.o: %.cu
	nvcc -dc $< -o $@

clean:
	rm -f *.o thrust.exe

.PHONY: clean all
