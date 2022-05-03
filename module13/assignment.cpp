//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char **argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id *platformIDs;
    cl_device_id *deviceIDs;

    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> commandQueues;
    std::vector<cl_event> events;
    std::vector<int *> inputOutputs;
    std::vector<cl_mem> buffers;

    cl_int scalar = atoi(argv[1]);

    std::vector<char *> kernelNames;
    for (int i = 2; i < argc; i++)
    {
        kernelNames.push_back(argv[i]);
    }

    int platform = DEFAULT_PLATFORM;

    std::cout << "Events and queues example" << std::endl;

    // First, select an OpenCL platform to run on.
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    platformIDs = (cl_platform_id *)alloca(
        sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl;

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char *src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform],
        CL_PLATFORM_VENDOR,
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices,
        &deviceIDs[0],
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platformIDs[platform],
            0};

    for (int i = 0; i < kernelNames.size(); i++)
    {
        contexts.push_back(clCreateContext(
            contextProperties,
            numDevices,
            deviceIDs,
            NULL,
            NULL,
            &errNum));
        checkErr(errNum, "clCreateContext");

        programs.push_back(clCreateProgramWithSource(
            contexts[i],
            1,
            &src,
            &length,
            &errNum));
        checkErr(errNum, "clCreateProgramWithSource");

        errNum = clBuildProgram(
            programs[i],
            numDevices,
            deviceIDs,
            "-I.",
            NULL,
            NULL);

        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(
                programs[i],
                deviceIDs[0],
                CL_PROGRAM_BUILD_LOG,
                sizeof(buildLog),
                buildLog,
                NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
        }

        buffers.push_back(clCreateBuffer(
            contexts[i],
            CL_MEM_READ_WRITE,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            NULL,
            &errNum));
        checkErr(errNum, "clCreateBuffer");

        commandQueues.push_back(clCreateCommandQueue(
            contexts[i],
            deviceIDs[0],
            0,
            &errNum));
        checkErr(errNum, "clCreateCommandQueue");

        kernels.push_back(clCreateKernel(
            programs[i],
            kernelNames[i],
            &errNum));
        checkErr(errNum, "clCreateKernel(square)");

        errNum = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg(arg1)");

        // cl_int scalar = i + 1;
        errNum = clSetKernelArg(kernels[i], 1, sizeof(cl_int), (void *)&scalar);

        int *inputOutput = new int[NUM_BUFFER_ELEMENTS];
        for (unsigned int j = 0; j < NUM_BUFFER_ELEMENTS; j++)
        {
            inputOutput[j] = j;
        }

        inputOutputs.push_back(inputOutput);

        errNum = clEnqueueWriteBuffer(
            commandQueues[i],
            buffers[i],
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS,
            (void *)inputOutputs[i],
            0,
            NULL,
            NULL);
        checkErr(errNum, "clEnqueueWriteBuffer");

        cl_event event;
        clEnqueueMarker(commandQueues[i], &event);
        events.push_back(event);
    }

    size_t gWI = NUM_BUFFER_ELEMENTS;

    for (int i = 0; i < kernels.size(); i++)
    {
        clEnqueueNDRangeKernel(
            commandQueues[i],
            kernels[i],
            1,
            NULL,
            (const size_t *)&gWI,
            (const size_t *)NULL,
            0,
            0,
            &events[i]);

        clEnqueueBarrier(commandQueues[i]);
        if (i < kernels.size() - 1)
        {
            clEnqueueWaitForEvents(commandQueues[i], 1, &events[i + 1]);
        }
    }

    for (int i = 0; i < kernels.size(); i++)
    {
        // reading back data
        clEnqueueReadBuffer(
            commandQueues[i],
            buffers[i],
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS,
            (void *)inputOutputs[i],
            0,
            NULL,
            NULL);
    }

    for (int i = 0; i < inputOutputs.size(); i++)
    {
        for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
        {
            std::cout << " " << inputOutputs[i][elems];
        }
        std::cout << std::endl;
    }

    cl_ulong time_start;
    cl_ulong time_end;

    // we process the last input first and the first input last
    clGetEventProfilingInfo(events[events.size() - 1], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end - time_start;
    std::cout << "OpenCL execution time is: " << nanoSeconds / 1000000.0 << "ms" << std::endl;

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
