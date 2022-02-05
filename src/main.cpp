#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <ctime>

using namespace std;

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *gpu_inputImageRGBA__;
uchar4 *gpu_outputImageRGBA__;

unsigned char *gpu_red, *gpu_green, *gpu_blue;
float         *gpu_filter;


void kernel(const uchar4 * const host_inputImageRGBA, uchar4 * const gpu_inputImageRGBA,
                        uchar4* const gpu_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *gpu_redBlurred,
                        unsigned char *gpu_greenBlurred,
                        unsigned char *gpu_blueBlurred,
                        const int filterWidth);
                
size_t getRowsCount() { 
  return imageInputRGBA.rows; 
}
size_t getColsCount() { 
  return imageInputRGBA.cols; 
}


int main(int argc, char **argv) {

  uchar4 *host_inputImageRGBA,  *gpu_inputImageRGBA;
  uchar4 *host_outputImageRGBA, *gpu_outputImageRGBA;

  unsigned char *gpu_redBlurred, *gpu_greenBlurred, *gpu_blueBlurred;
  float *host_filter;
  int    filterWidth;

  string input_file="images/flower_300x300.jpg";
  string output_file="images/iesire_gpu.jpg";
  
  if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
  }
  else {
    cout << "Input file or output file not specified. Using default ones" << endl;
  }

  // init context in the GPU
  cudaFree(0);

  cv::Mat image = cv::imread(input_file.c_str());
  if (image.empty()) {
    cerr << "Couldn't open file: " << input_file << endl;
    exit(1);
  }

  cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);//conversion code to RGBA

  //allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  host_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  host_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

  const size_t image_size_pixels = getRowsCount()*getColsCount();
   
  //create a 5x5 Gaussian filter which is a Low-Pass Filter 
  //standard deviation 4 
  // OpenCV comparsion (with same 5x5 filter and sigma 4)
  const int gaussKernelWidth = 5;
  const float gaussKernelSigma = 4.;

  filterWidth = gaussKernelWidth;

  //create and fill the filter we will convolve with
  host_filter = new float[gaussKernelWidth * gaussKernelWidth];

  float filterSum = 0.f; //for normalization

  for (int r = -gaussKernelWidth/2; r <= gaussKernelWidth/2; ++r) {
    for (int c = -gaussKernelWidth/2; c <= gaussKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * gaussKernelSigma * gaussKernelSigma));
      (host_filter)[(r + gaussKernelWidth/2) * gaussKernelWidth + c + gaussKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -gaussKernelWidth/2; r <= gaussKernelWidth/2; ++r) {
    for (int c = -gaussKernelWidth/2; c <= gaussKernelWidth/2; ++c) {
      (host_filter)[(r + gaussKernelWidth/2) * gaussKernelWidth + c + gaussKernelWidth/2] *= normalizationFactor;
    }
  }


  auto start = std::chrono::high_resolution_clock::now();
  //allocate memory on the device for both input and output
  cudaMalloc(&gpu_inputImageRGBA, sizeof(uchar4) * image_size_pixels);
  cudaMalloc(&gpu_outputImageRGBA, sizeof(uchar4) * image_size_pixels);

  //clean the allocated memory (simillar to calloc)
  cudaMemset(gpu_outputImageRGBA, 0, sizeof(uchar4) * image_size_pixels ); 

  //copy input array to the GPU
  cudaMemcpy(gpu_inputImageRGBA, host_inputImageRGBA, sizeof(uchar4) * image_size_pixels, cudaMemcpyHostToDevice);

  //make a copy of the input and output image 
  gpu_inputImageRGBA__  = gpu_inputImageRGBA;
  gpu_outputImageRGBA__ = gpu_outputImageRGBA;

  //allocate memory for destination images 
  //for all three channels RGB
  cudaMalloc(&gpu_redBlurred,    sizeof(unsigned char) * image_size_pixels);
  cudaMalloc(&gpu_greenBlurred,  sizeof(unsigned char) * image_size_pixels);
  cudaMalloc(&gpu_blueBlurred,   sizeof(unsigned char) * image_size_pixels);

  cudaMemset(gpu_redBlurred,   0, sizeof(unsigned char) * image_size_pixels);
  cudaMemset(gpu_greenBlurred, 0, sizeof(unsigned char) * image_size_pixels);
  cudaMemset(gpu_blueBlurred,  0, sizeof(unsigned char) * image_size_pixels);

 

  //work with with RGB in CUDA (without ALPHA channel)
  cudaMalloc(&gpu_red,   sizeof(unsigned char) * image_size_pixels);
  cudaMalloc(&gpu_green, sizeof(unsigned char) * image_size_pixels);
  cudaMalloc(&gpu_blue,  sizeof(unsigned char) * image_size_pixels);

  
  //Allocate mem for filter on GPU
  cudaMalloc(&gpu_filter, sizeof(float) * filterWidth * filterWidth);

  //copy filter on GPU 
  cudaMemcpy(gpu_filter, host_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice);

  kernel(host_inputImageRGBA, gpu_inputImageRGBA, gpu_outputImageRGBA, getRowsCount(), getColsCount(), gpu_redBlurred, gpu_greenBlurred, gpu_blueBlurred, filterWidth);
  
  //cudaDeviceSynchronize(); 

  //copy the output back to the host
  cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), gpu_outputImageRGBA__, sizeof(uchar4) * image_size_pixels, cudaMemcpyDeviceToHost);
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "GPU: Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << " ms" << endl;
  //convert the image back
  cv::Mat imageOutputBGR;
  cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);

  //cleanup
  delete [] host_filter;
  cudaFree(gpu_inputImageRGBA__);
  cudaFree(gpu_outputImageRGBA__);
  cudaFree(gpu_redBlurred);
  cudaFree(gpu_greenBlurred);
  cudaFree(gpu_blueBlurred);

  cudaFree(gpu_red);
  cudaFree(gpu_green);
  cudaFree(gpu_blue);

  cudaFree(gpu_filter);

  return 0;
}
