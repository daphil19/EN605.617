objects = operations.o assignment.o

all: assignment.exe stretch.exe

assignment.exe: $(objects)
	nvcc $(objects) -lcudart -o assignment.exe

%.o: %.cu
	nvcc -dc $< -o $@

stretch.exe: stretch_problem_module4.cu
	nvcc $< -lcudart -o $@

clean:
	rm -f *.o assignment.exe stretch.exe

.PHONY: clean all