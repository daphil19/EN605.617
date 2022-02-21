objects = operations.o assignment.o

all: $(objects)
	nvcc $(objects) -lcudart -o assignment.exe

%.o: %.cu
	nvcc -dc $< -o $@

clean:
	rm -f *.o assignment.exe

.PHONY: clean all