objects = assignment.o

all: assignment.exe

assignment.exe: $(objects)
	nvcc $(objects) -lcudart -lcufft -lcublas -o assignment.exe

%.o: %.cu
	nvcc -dc $< -o $@

clean:
	rm -f *.o assignment.exe

.PHONY: clean all