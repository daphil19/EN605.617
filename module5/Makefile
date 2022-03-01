objects = assignment.o

all: assignment.exe

assignment.exe: $(objects)
	nvcc $(objects) -lcudart -o assignment.exe

%.o: %.cu
	nvcc -dc $< -o $@

clean:
	rm -f *.o assignment.exe

.PHONY: clean all