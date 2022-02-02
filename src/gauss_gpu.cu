
extern "C" unsigned char *gpu_red, *gpu_green, *gpu_blue;
extern "C" float         *gpu_filter;

__global__ void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{

  //define iterators 
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;

  //verify out of bound conditions
  if (px >= numCols || py >= numRows) {
      return;
  }

  float c = 0.0f;
  
  //perform convolution
  for (int fx = 0; fx < filterWidth; fx++) {
    for (int fy = 0; fy < filterWidth; fy++) {
      int imagex = px + fx - filterWidth / 2;
      int imagey = py + fy - filterWidth / 2;
      imagex = min(max(imagex,0),numCols-1);
      imagey = min(max(imagey,0),numRows-1);
      c += (filter[fy*filterWidth+fx] * inputChannel[imagey*numCols+imagex]);
    }
  }

  outputChannel[py*numCols+px] = c;
}

// split into rgb channels
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{

  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= numCols || py >= numRows) {
      return;
  }
  int i = py * numCols + px;
  redChannel[i] = inputImageRGBA[i].x;
  greenChannel[i] = inputImageRGBA[i].y;
  blueChannel[i] = inputImageRGBA[i].z;
}

// combine color channels
__global__ void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  // check out-of-bounds
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //255 is for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}


void kernel(const uchar4 * const host_inputImageRGBA, uchar4 * const gpu_inputImageRGBA,
                        uchar4* const gpu_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *gpu_redBlurred, 
                        unsigned char *gpu_greenBlurred, 
                        unsigned char *gpu_blueBlurred,
                        const int filterWidth)
{
  //set threads per block
  const dim3 blockSize(16,16,1);

  //calc grid size
  const dim3 gridSize(numCols/blockSize.x+1,numRows/blockSize.y+1,1);

  //separate color channels
  separateChannels<<<gridSize, blockSize>>>(gpu_inputImageRGBA,numRows,numCols,gpu_red,gpu_green,gpu_blue);

  //cudaDeviceSynchronize();

  // call blur for each channel (RGB)
  gaussian_blur<<<gridSize, blockSize>>>(gpu_red,gpu_redBlurred,numRows,numCols,gpu_filter,filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(gpu_green,gpu_greenBlurred,numRows,numCols,gpu_filter,filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(gpu_blue,gpu_blueBlurred,numRows,numCols,gpu_filter,filterWidth);

  cudaDeviceSynchronize(); 

  // recombine channels to final image
  recombineChannels<<<gridSize, blockSize>>>(gpu_redBlurred,
                                             gpu_greenBlurred,
                                             gpu_blueBlurred,
                                             gpu_outputImageRGBA,
                                             numRows,
                                             numCols);
  //cudaDeviceSynchronize(); 

}



