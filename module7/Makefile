objects = assignment.o operations.o

all: assignment.exe stretch.exe

assignment.exe: $(objects)
	nvcc $(objects) -lcudart -o assignment.exe

%.o: %.cu
	nvcc -dc $< -o $@

stretch.exe: module7_stretch_problem.cu
	nvcc -lcudart $< -o $@


clean:
	rm -f *.o assignment.exe stretch.exe

.PHONY: clean all