#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/cudev/functional/functional.hpp>
#include <thrust/complex.h>
#include "render.cuh"



__device__ thrust::complex<double> f1(thrust::complex<double> z, double time)
{
  using namespace thrust;
  using namespace cv::cudev;

  z /= complex<double>(141,0);
  z *= complex<double>(0.35*(abs(abs(sin(time / (1000*M_PI) / 360)) - 0.5)) + 0.7,0);
  z = pow(z, pow(z, pow(z,pow(z,z))));
  z *= complex<double>(sin(time / (2000*M_PI) / 1), cos(time / (2000*M_PI) / 1));

  return z;
}

__device__ thrust::complex<double> f2(thrust::complex<double> z, double time)
{
  using namespace thrust;
  using namespace cv::cudev;

  z /= complex<double>(500,0);
  z = complex<double>(1,0) / z;
  z = complex<double>(1/z.real(), 1/z.imag());
  z = complex<double>(-abs(z.real()), z.imag());
  z = pow(z, z);

  return z;
}

__device__ thrust::complex<double> f3(thrust::complex<double> z, double time)
{
  using namespace thrust;
  using namespace cv::cudev;
  z /= complex<double>(200,0);

  z = pow(z,2) - complex<double>(1,0);
  z *= pow(z - complex<double>(2,1), 2);
  z /= pow(z,2) + complex<double>(2,2);
  // double k = abs(((intmax_t)round(time/10) % 5000) - 2500)/100.0;
  double k = 10.8; // 0 - 25
  z = complex<double>(round(z.real()*k), round(z.imag()*k));

  return z;
}

__device__ thrust::complex<double> f4(thrust::complex<double> z, double time)
{
  using namespace thrust;
  using namespace cv::cudev;

  z /= complex<double>(10,0);
  z = complex<double>(fmod(z.real(),z.imag()), fmod(z.imag(), z.real()));
  z = tanh(sinh(z));

  return z;
}


__device__ thrust::complex<double> f(thrust::complex<double> z, double time, uintmax_t mode)
{
  using namespace thrust;
  using namespace cv::cudev;

  if(mode == 0){ //animacja
    complex<double> z1 = f1(z, time); // z^z^z^z
    complex<double> z2 = f2(z, time); // 1/x
    return z1 + z2;
  }
  else if(mode == 1){ //flower
    complex<double> z3 = f3(z, time); //flower
    return z3;
  }
  else if(mode == 2){
    complex<double> z4 = f4(z, time); // towers
    return z4;
  }

  return z;
}







#define NUM_THREADS 768
#define CHANNELS 3

using namespace std;
using namespace cv;
using namespace cv::cuda;

__global__ void renderKernel(GpuMat frame, double time, uintmax_t mode)
{
  const uintmax_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uintmax_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x < frame.cols) && (y < frame.rows)){
		const uintmax_t i = y * frame.step + (CHANNELS * x);

    double X = (double)x - (frame.cols/2);
    double Y = (double)y - (frame.rows/2);
    double k = 1920/((double)frame.cols);
    X *= k;
    Y *= k;
    thrust::complex<double> z(X,Y);

    z = f(z, time, mode);

    X = z.real();
    Y = z.imag();

    double H = cudev::atan2_func<double>()(Y, X) * (180.0/M_PI);
    if(H < 0)
      H += 360.0;

    double L = (1 - cudev::pow_func<double>()(0.75, cudev::sqrt_func<double>()(cudev::pow_func<double>()(X,2)+cudev::pow_func<double>()(Y,2)))) * 255.0;
    if(L > 255)
      L = 255;

    //HSL
		frame.data[i + 0] = (uint8_t)round(H/2);
    frame.data[i + 1] = (uint8_t)round(L);
    frame.data[i + 2] = 255;
	}
}

extern "C" void render(GpuMat frame, double time, uintmax_t mode)
{
  const dim3 block(16,16);
  const dim3 grid(cudev::divUp(frame.cols, block.x), cudev::divUp(frame.rows, block.y));
  renderKernel<<<grid, block>>>(frame, time, mode);
  cudaDeviceSynchronize();
}
