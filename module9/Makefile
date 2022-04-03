thrust_objects = thrust.o
npp_and_nvgraph_objects = npp_and_nvgraph.o

all: thrust.exe npp_and_nvgraph.exe

thrust.exe: $(thrust_objects)
	nvcc $< -o $@

npp_and_nvgraph.exe: $(npp_and_nvgraph_objects)
	nvcc $< -lnppi_static -o $@

%.o: %.cu
	nvcc -dc $< -o $@

clean:
	rm -f *.o thrust.exe npp_and_nvgraph.exe

.PHONY: clean all
