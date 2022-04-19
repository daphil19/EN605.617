//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const int MAX_RANDOM = 10;

template <size_t signalHeight, size_t signalWidth>
void createSignal(cl_uint (&signal)[signalHeight][signalWidth]) {
	for (int i = 0; i < signalHeight; i++) {
		for (int j = 0; j < signalWidth; j++) {
			signal[i][j] = rand() % MAX_RANDOM;
		}
	}
}

void foo(cl_uint **signal) {

}

template <size_t maskHeight, size_t maskWidth>
void createMask(cl_uint (&mask)[maskHeight][maskWidth]) {
	// this logic assumes that dimensions will always be odd
	// (and possibly with a corrolary that they're square)
	// so that there is a true center
	int centerRow = maskHeight / 2;
	int centerCol = maskWidth / 2;

	for (int i = 0; i < maskHeight; i++) {
		for (int j = 0; j < maskWidth; j++) {
			int rowDist = abs(centerRow - i);
			int colDist = abs(centerCol - j);

			// the highest intensity is at the center (center intex + 1)
			// each "unit" distance away has a lower insensity, until the outer
			// perimiter, which has a value of 1
			// determining the "distance" is simlpy the larger of the row and
			// column distances (this is because we are using a box)
			mask[i][j] = centerRow + 1 - std::max(rowDist, colDist);
		}
	}
}

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

template <size_t size>
void print(cl_uint (&data)[size][size]) {
	for (int y = 0; y < size; y++)
	{
		for (int x = 0; x < size; x++)
		{
			std::cout << data[y][x] << " ";
		}
		std::cout << std::endl;
	}
}

template <size_t signalSize, size_t maskSize>
void performFilter(cl_context &context, cl_kernel &kernel, cl_command_queue &queue, cl_device_id * deviceIDs) {
	cl_int errNum;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

	cl_uint inputSignal[signalSize][signalSize];
	cl_uint mask[maskSize][maskSize];

	createSignal<signalSize, signalSize>(inputSignal);
	
	std::cout << "signal:" << std::endl;
	print<signalSize>(inputSignal);
	
	createMask<maskSize, maskSize>(mask);

	std::cout << "mask:" << std::endl;
	print<maskSize>(mask);

	const unsigned int outputSignalSize = signalSize - maskSize + 1;

	cl_uint outputSignal[outputSignalSize][outputSignalSize];

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * signalSize * signalSize,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskSize * maskSize,
		static_cast<void *>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalSize * outputSignalSize,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		CL_QUEUE_PROFILING_ENABLE,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

	// needed to create these so that their addresses could be passed in to the kernel
	size_t ss = signalSize;
	size_t ms = maskSize;

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &ss);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &ms);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalSize, outputSignalSize };
    const size_t localWorkSize[2]  = { 1, 1 };

	cl_event event;

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		&event);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
	clWaitForEvents(1, &event);
	clFinish(queue);

	errNum = clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalSize * outputSignalSize, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

    // Output the result buffer
	std::cout << "output:" << std::endl;
	print<outputSignalSize>(outputSignal);

	cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double nanoSeconds = time_end-time_start;
    std::cout << "OpenCL execution time is: " << nanoSeconds / 1000000.0 << "ms" << std::endl;

	// cleanup
	clReleaseMemObject(inputSignalBuffer);
	clReleaseMemObject(maskBuffer);
	clReleaseMemObject(outputSignalBuffer);
}

///
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		CL_QUEUE_PROFILING_ENABLE,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

	performFilter<8, 3>(context, kernel, queue, deviceIDs);

	performFilter<49, 7>(context, kernel, queue, deviceIDs);

	// cleanup
	clReleaseCommandQueue(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);

    std::cout << std::endl << "Executed program succesfully." << std::endl;

	return 0;
}
