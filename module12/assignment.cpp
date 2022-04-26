#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

// globals/#defines
#define NUM_BUFFER_ELEMENTS 16
#define SUB_BUFFER_SIZE 4
#define NUM_SUB_BUFFERS NUM_BUFFER_ELEMENTS / SUB_BUFFER_SIZE

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

void createProgram(int &platform, cl_uint &numDevices, cl_platform_id *platformIDs, cl_device_id *deviceIDs, cl_context &context, cl_program &program) {
cl_int errNum;

        std::ifstream srcFile("filter.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading filter.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char *src = srcProg.c_str();
    size_t length = srcProg.length();

        cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platformIDs[platform],
            0};

    context = clCreateContext(
        contextProperties,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateContext");

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
        "-I.",
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

        std::cerr << "Error in OpenCL C source: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
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
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    int *inputOutput;

    int platform = 0;

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

    createProgram(platform, numDevices, platformIDs, deviceIDs, context, program);

    inputOutput = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput[i] = i;
    }

    // create a single buffer to cover all the input data
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    for (unsigned int i = 0; i < NUM_SUB_BUFFERS; i++)
    {
        // here, we create sub-buffers that each take a portion of the main buffer
        cl_buffer_region region =
            {
                4 * i * sizeof(int),
                4 * sizeof(int)};
        cl_mem buffer = clCreateSubBuffer(
            main_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }

    // Create command queues
    for (unsigned int i = 0; i < NUM_SUB_BUFFERS; i++)
    {
        InfoDevice<cl_device_type>::display(
            deviceIDs[0],
            CL_DEVICE_TYPE,
            "CL_DEVICE_TYPE");

        cl_command_queue queue =
            clCreateCommandQueue(
                context,
                deviceIDs[0],
                CL_QUEUE_PROFILING_ENABLE,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "filter",
            &errNum);
        checkErr(errNum, "clCreateKernel(filter)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg(filter arg 0)");

        cl_int bufferSize = SUB_BUFFER_SIZE;
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&bufferSize);
        checkErr(errNum, "clSetKernelArg(filter arg 1)");

        kernels.push_back(kernel);
    }

    errNum = clEnqueueWriteBuffer(
        queues[numDevices - 1],
        main_buffer,
        CL_TRUE,
        0,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        (void *)inputOutput,
        0,
        NULL,
        NULL);

    std::vector<cl_event> events;
    // call kernel for each device
    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = NUM_BUFFER_ELEMENTS;

        errNum = clEnqueueNDRangeKernel(
            queues[i],
            kernels[i],
            1,
            NULL,
            (const size_t *)&gWI,
            (const size_t *)NULL,
            0,
            0,
            &event);

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

	cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(events[events.size() - 1], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);


    // Read back computed data
    clEnqueueReadBuffer(
        queues[numDevices - 1],
        main_buffer,
        CL_TRUE,
        0,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        (void *)inputOutput,
        0,
        NULL,
        NULL);

    // Display output in rows
    for (unsigned i = 0; i < numDevices; i++)
    {
        for (unsigned elems = i * NUM_BUFFER_ELEMENTS; elems < ((i + 1) * NUM_BUFFER_ELEMENTS); elems++)
        {
            std::cout << " " << inputOutput[elems];
        }

        std::cout << std::endl;
    }

    double nanoSeconds = time_end-time_start;
    std::cout << "OpenCL execution time is: " << nanoSeconds / 1000000.0 << "ms" << std::endl;

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}