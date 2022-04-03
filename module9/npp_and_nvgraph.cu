// #include <string>
// #include <iostream>
// #include <fstream>

// #include <npp.h>

// int main(int argc, char const *argv[])
// {
//     // TODO get from cli args
//     std::string filename;

//     std::ifstream infile(filename.data(), std::ifstream::in);

//     if (!infile.good()) {
//         std::cout << "unable to open file: <" << filename.data() << ">" << std::endl;
//     }

//     std::string outFilename = filename;

//     std::string::size_type dot = outFilename.rfind('.');

//     if (dot != std::string::npos)
//     {
//         outFilename = outFilename.substr(0, dot);
//     }

//     // TODO rename the extension!
//     outFilename += "_boxFilter.pgm";


//     // TODO we need an image handle
//     // npp:

//     return EXIT_SUCCESS;
// }

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

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
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

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

        // create struct with box-filter mask size
        NppiSize oMaskSize = {5, 5};

        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};


        double imageScaling = 2;

        // create struct with ROI size
        // NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
        NppiRect oSrcROI = {0, 0, oDeviceSrc.width(), oDeviceSrc.height()};
        // allocate device image of appropriately reduced size
        // std::cout << "crating dst " << oSrcROI.width * imageScaling << " " << oSrcROI.height * imageScaling << std::endl;
        npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.width() * imageScaling, oDeviceSrc.height() * imageScaling);
        std::cout << "done" << std::endl;
        // set anchor point inside the mask to (oMaskSize.width / 2, oMaskSize.height / 2)
        // It should round down when odd
        NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

        NppiSize dstROISize = {oDeviceDst.width(), oDeviceDst.height()};

        /*NppStatus nppiResize_8u_C1R(
            const Npp8u *pSrc, 
            NppiSize oSrcSize, 
            int nSrcStep, 
            NppiRect oSrcROI, 
            Npp8u *pDst, 
            int nDstStep, 
            NppiSize dstROISize, 
            double nXFactor, 
            double nYFactor, 
            int eInterpolation
        )*/

        // nppiResize_8u_C1R(
        //     oDeviceSrc.data(), 
        //     oSrcSize, 
        //     oDeviceSrc.pitch(), 
        //     oSizeROI,
        //     oDeviceDst.data(),
        //     oDeviceDst.pitch(),
        //     dstROISize,
        //     imageScaling,
        //     imageScaling,
        //     NPPI_INTER_CUBIC
        // );

        /*NppStatus nppiFilterBoxBorder_8u_C1R(
            const Npp8u *pSrc,
            Npp32s nSrcStep, 
            NppiSize oSrcSize,
            NppiPoint oSrcOffset, 
            Npp8u *pDst, 
            Npp32s nDstStep, 
            NppiSize oSizeROI, 
            NppiSize oMaskSize, 
            NppiPoint oAnchor, 
            NppiBorderType eBorderType
        )*/


        // run box filter
        // NPP_CHECK_NPP (
        //                    nppiFilterBoxBorder_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
        //                                               oSrcSize, oSrcOffset,
        //                                               oDeviceDst.data(), oDeviceDst.pitch(),
        //                                               oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE) );

        // std::cout << oDeviceSrc.width() << std::endl <<  oDeviceSrc.height() << std::endl;

        NPP_CHECK_NPP (
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
                NPPI_INTER_CUBIC
            )
        );

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        exit(EXIT_SUCCESS);
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
        return -1;
    }

    return 0;
}
