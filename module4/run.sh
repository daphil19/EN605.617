#!/bin/bash
make
echo "Running iterations to verify correctness..."
./assignment.exe
./assignment.exe 16384 1024
./assignment.exe 512 256
echo "Running Caesar cypher code..."
./stretch.exe 8960 256 text.txt key.txt
echo "Running iterations to measure performance..."
./assignment.exe 1048576 256 --quiet
./assignment.exe 16384 1024 --quiet
./assignment.exe 512 256 --quiet