all: HelloWorld.exe

HelloWorld.exe: HelloWorld.cpp
# use of nvcc for compilation because cuda provides OpenCL libs as well
	nvcc -lOpenCL $< -o $@

clean:
	rm -f HelloWorld.exe

.PHONY: clean all
