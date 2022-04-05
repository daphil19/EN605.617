///////////////////////////////////////////////////////////////////////////////
// Portions of this source code were adapted from both the nvgraph_Pagerank and
// boxFilterNPP examples found in this repo
///////////////////////////////////////////////////////////////////////////////

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <npp.h>
#include <nvgraph.h>

#include <helper_string.h>
#include <helper_cuda.h>

__host__ cudaEvent_t get_time(void)
{
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
    return time;
}

// print the delta based on the provided start and stop events
__host__ void print_delta(cudaEvent_t start, cudaEvent_t stop)
{
    cudaEventSynchronize(stop);

    float delta = 0;
    cudaEventElapsedTime(&delta, start, stop);
    printf("%f\n", delta);
}

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

void perform_npp_operation(int argc, char *argv[])
{
    try
    {
        std::string sFilename;
        char *filePath;

        cudaDeviceInit(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        double imageScaling;

        if (argc >= 2)
        {
            imageScaling = atof(argv[1]);
            std::cout << "Image will be scaled by a factor of " << imageScaling << " in both directions." << std::endl;
        }
        else
        {
            imageScaling = 2;
            std::cout << "No image scaling argument provided. Image will be scaled by a default factor of " << imageScaling << std::endl;
        }

        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_resized.pgm";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        NppiRect oSrcROI = {0, 0, oDeviceSrc.width(), oDeviceSrc.height()};
        // allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.width() * imageScaling, oDeviceSrc.height() * imageScaling);
        std::cout << "done" << std::endl;

        NppiSize dstROISize = {oDeviceDst.width(), oDeviceDst.height()};

        NPP_CHECK_NPP(
            nppiResize_8u_C1R(
                oDeviceSrc.data(),
                oSrcSize,
                oDeviceSrc.pitch(),
                oSrcROI,
                oDeviceDst.data(),
                oDeviceDst.pitch(),
                dstROISize,
                imageScaling,
                imageScaling,
                NPPI_INTER_CUBIC));

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
}

void perform_nvgraph_operation()
{
    printf("Performing nvgraph operation...\n");
    const size_t num_vertices = 6, num_edges = 10, vertex_numsets = 1, edge_numsets = 1;
    int i, *destination_offsets, *source_indices;
    float *weights;
    void **vertex_dim;

    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t topology;
    cudaDataType_t edge_dim_t = CUDA_R_32F;
    cudaDataType_t *vertex_dim_t;

    destination_offsets = new int[num_vertices + 1];
    source_indices = new int[num_edges];
    weights = new float[num_edges];
    vertex_dim = new void *[vertex_numsets];
    vertex_dim_t = new cudaDataType_t[vertex_numsets];
    topology = new nvgraphCSCTopology32I_st;

    float *shortest_path_res = new float[num_vertices];

    vertex_dim[0] = (void *)shortest_path_res;
    vertex_dim_t[0] = CUDA_R_32F;

    weights[0] = 0.333333f;
    weights[1] = 0.500000f;
    weights[2] = 0.333333f;
    weights[3] = 0.500000f;
    weights[4] = 0.500000f;
    weights[5] = 1.000000f;
    weights[6] = 0.333333f;
    weights[7] = 0.500000f;
    weights[8] = 0.500000f;
    weights[9] = 0.500000f;

    destination_offsets[0] = 0;
    destination_offsets[1] = 1;
    destination_offsets[2] = 3;
    destination_offsets[3] = 4;
    destination_offsets[4] = 6;
    destination_offsets[5] = 8;
    destination_offsets[6] = 10;

    source_indices[0] = 2;
    source_indices[1] = 0;
    source_indices[2] = 2;
    source_indices[3] = 0;
    source_indices[4] = 4;
    source_indices[5] = 5;
    source_indices[6] = 2;
    source_indices[7] = 3;
    source_indices[8] = 3;
    source_indices[9] = 4;

    nvgraphCreate(&handle);
    nvgraphCreateGraphDescr(handle, &graph);

    topology->nvertices = num_vertices;
    topology->nedges = num_edges;
    topology->destination_offsets = destination_offsets;
    topology->source_indices = source_indices;

    nvgraphSetGraphStructure(handle, graph, (void *)topology, NVGRAPH_CSC_32);
    nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dim_t);
    nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dim_t);
    nvgraphSetEdgeData(handle, graph, (void *)weights, 0);

    int source_vertex = 0;

    nvgraphSssp(handle, graph, 0, &source_vertex, 0);

    // Get and print result
    nvgraphGetVertexData(handle, graph, (void *)shortest_path_res, 0);
    printf("Sssp result:\n");
    for (i = 0; i < num_vertices; i++)
        printf("%f\n", shortest_path_res[i]);
    printf("\n");

    // Clean
    nvgraphDestroyGraphDescr(handle, graph);
    nvgraphDestroy(handle);

    delete[] destination_offsets;
    delete[] source_indices;
    delete[] weights;
    delete[] vertex_dim;
    delete[] vertex_dim_t;
    delete topology;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    cudaEvent_t nppStart = get_time();
    perform_npp_operation(argc, argv);
    cudaEvent_t nppStop = get_time();

    cudaEvent_t nvgraphStart = get_time();
    perform_nvgraph_operation();
    cudaEvent_t nvgraphStop = get_time();

    std::cout << "NPP Operation: ";
    print_delta(nppStart, nppStop);

    std::cout << "nvGRAPH Operation: ";
    print_delta(nvgraphStart, nvgraphStop);

    return EXIT_SUCCESS;
}
