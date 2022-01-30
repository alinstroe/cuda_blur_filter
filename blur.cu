

__global__ void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{


  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= numCols || py >= numRows) {
      return;
  }

  float c = 0.0f;

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
__global__
void separateChannels(const uchar4* const inputImageRGBA,
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
__global__
void recombineChannels(const unsigned char* const redChannel,
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

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //mem for 3 mem channels
  cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);

  
  //Allocate mem for filter
  cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth);

  //copy filter on host
  cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice);
}

void kernel(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //set threads per block
  const dim3 blockSize(16,16,1);

  //calc grid size
  const dim3 gridSize(numCols/blockSize.x+1,numRows/blockSize.y+1,1);

  //separate color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,numRows,numCols,d_red,d_green,d_blue);

  cudaDeviceSynchronize();

  // call blur for each channel
  gaussian_blur<<<gridSize, blockSize>>>(d_red,d_redBlurred,numRows,numCols,d_filter,filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green,d_greenBlurred,numRows,numCols,d_filter,filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue,d_blueBlurred,numRows,numCols,d_filter,filterWidth);

  cudaDeviceSynchronize(); 

  // recombine channels to final image
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); 
}



