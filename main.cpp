#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <ctime>
using namespace std;

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }


void kernel(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth);


int main(int argc, char **argv) {

  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

  float *h_filter;
  int    filterWidth;

  string filename="flower_300x300.jpg";
  string output_file="iesire.jpg";
  /*
  if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
  }
  else {
    std::cerr << "Usage: ./filter input_file output_file" << std::endl;
    exit(1);
  }
  */
  cudaFree(0);

  cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "Couldn't open file: " << filename << endl;
    exit(1);
  }

  cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);//conversion code to RGBA

  //allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
 
  //allocate memory on the device for both input and output
  cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels);
  cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels);
  //clean the input
  cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4));

  //Transfer Data
  //copy input array to the GPU
  cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

  d_inputImageRGBA__  = d_inputImageRGBA;
  d_outputImageRGBA__ = d_outputImageRGBA;
  
  //now create the filter that they will use, orig 9 and 2
  const int blurKernelWidth = 5;
  const float blurKernelSigma = 4.;

  filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  h_filter = new float[blurKernelWidth * blurKernelWidth];

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  //allocate memory for destination images 
  //for all three channels RGB
  cudaMalloc(&d_redBlurred,    sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_greenBlurred,  sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_blueBlurred,   sizeof(unsigned char) * numPixels);

  cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * numPixels);
  cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
  cudaMemset(d_blueBlurred,  0, sizeof(unsigned char) * numPixels);

  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
  kernel(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(), d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);

  cudaDeviceSynchronize(); 
 
  //copy the output back to the host
  cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << "Done in " << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() << " us" << endl;
  //convert the image back
  cv::Mat imageOutputBGR;
  cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);

  //cleanup
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);


  cudaFree(d_redBlurred);
  cudaFree(d_greenBlurred);
  cudaFree(d_blueBlurred);

  return 0;
}
