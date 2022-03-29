objects = assignment.o

all: assignment.exe

assignment.exe: $(objects)
	nvcc $(objects) -lcudart -lcufft -lcublas -lcurand -o assignment.exe

%.o: %.cu
	nvcc -dc $< -o $@

clean:
	rm -f *.o assignment.exe

.PHONY: clean all