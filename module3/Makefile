all: assignment.exe advanced-assignment

assignment.exe: assignment.cu
	nvcc assignment.cu -lcurand -o assignment.exe

advanced-assignment: advanced-assignment.cu
	nvcc -std=c++11 advanced-assignment.cu -lcurand -o advanced-assignment

clean:
	rm assignment.exe advanced-assignment

.PHONY: clean all