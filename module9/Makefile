thrust_objects = thrust.o
npp_and_nvgraph_objects = npp_and_nvgraph.o

all: thrust.exe npp_and_nvgraph.exe

thrust.exe: $(thrust_objects)
	nvcc $< -o $@

npp_and_nvgraph.exe: $(npp_and_nvgraph_objects)
	nvcc $< -lnppi_static -o $@ -L./common/FreeImage/lib/ -L./common/FreeImage/lib/linux -L./common/FreeImage/lib/linux/x86_64 -lnppi_static -lnppc_static -lculibos -lfreeimage

%.o: %.cu
	nvcc -dc $< -o $@ -I./common/UtilNPP -I./common/FreeImage/include -I../common/inc

clean:
	rm -f *.o thrust.exe npp_and_nvgraph.exe

.PHONY: clean all
