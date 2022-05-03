all: assignment.exe
	
assignment.exe: assignment.cpp
# use of nvcc for compilation because cuda provides OpenCL libs as well
	nvcc -lOpenCL $< -o $@

clean:
	rm -f assignment.exe

.PHONY: all clean
