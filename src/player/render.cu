#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/cudev/functional/functional.hpp>
#include <thrust/complex.h>
#include "render.cuh"

using namespace std;
using namespace cv;
using namespace cv::cuda;

__global__ void renderKernel(GpuMat frame, uint8_t* values, uint8_t* repetitions, uint8_t* toBeLoaded)
{
    const uintmax_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uintmax_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x < frame.cols) && (y < frame.rows)){
      const uintmax_t i = y * frame.step + (3 * x);

      frame.data[i + 0] = values[3*((x*frame.rows)+y) + 0];
      frame.data[i + 1] = values[3*((x*frame.rows)+y) + 1];
      frame.data[i + 2] = values[3*((x*frame.rows)+y) + 2];

      if(repetitions[(x*frame.rows) + y] == 0)
        toBeLoaded[(x*frame.rows) + y] = 1;
      else
        repetitions[(x*frame.rows) + y] -= 1;
    }
}

extern "C" void render(GpuMat frame, uint8_t* values, uint8_t* repetitions, uint8_t* toBeLoaded)
{
  const dim3 block(16,16);
  const dim3 grid(cudev::divUp(frame.cols, block.x), cudev::divUp(frame.rows, block.y));
  renderKernel<<<grid, block>>>(frame, values, repetitions, toBeLoaded);
  cudaDeviceSynchronize();
}
