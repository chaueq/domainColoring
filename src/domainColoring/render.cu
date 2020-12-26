#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/cudev/functional/functional.hpp>
#include <thrust/complex.h>
#include "render.cuh"

#define M_PHI 1.618033988749895


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

__device__ thrust::complex<double> f5(thrust::complex<double> z, double time)
{
  using namespace thrust;
  using namespace cv::cudev;

  z /= complex<double>(100, 0);
  z = acosh(z) + cos(z);
  z = complex<double>(cos(z.real()), sin(z.imag()));
  double k = pow(M_PHI, M_PI); //0-25
  z = complex<double>(round(z.real()*k), round(z.imag()*k));
  z = complex<double>(z.real()/pow(M_PHI, 2), z.imag()/pow(M_PHI, 2));

  return z;
}

__device__ thrust::complex<double> f6(thrust::complex<double> z, double time)
{
  using namespace thrust;
  using namespace cv::cudev;

  z /= 500.0;

  if(norm(z) > 0.5)
  {
    z *= 500.0;
    complex<double> sum(0,0);
    intmax_t f[] = {0,1,2,4,6,11,18,31,54,97,172,309,564,1028,1900,
 3512,6542,12251,23000,43390,82025,155611,295947,
 564163,1077871,2063689,3957809,7603553,14630843,
 28192750,54400028,105097565,203280221,393615806,
 762939111,1480206279,2874398515,5586502348,
 10866266172,21151907950,41203088796,80316571436,
 156661034233,305761713237,597116381732,
 1166746786182,2280998753949,4461632979717,
 8731188863470,17094432576778,33483379603407,
 65612899915304,128625503610475};
    for(uintmax_t m = 0; m < 52; ++m)
      for(uintmax_t x = 1; x < 52; ++x)
      {
        double a = f[m]*f[x-1];
        double b = f[m]*f[x];
        sum += pow(abs(sin(b))/((z/2.0) - complex<double>((1.0-abs(cos(a)))*((intmax_t)a%960 - 480), (1.0-abs(sin(b)))*((intmax_t)b%540 - 270))),3);
      }
    return sum;
  }

  z = exp(z);
  z /= M_PHI;

  return z;
}

__device__ thrust::complex<double> f7(thrust::complex<double> z, double time)
{
  using namespace thrust;
  using namespace cv::cudev;

  z /= 50.0;
  z -= 13.0;
  z = 1.0/z;
  z = pow(z, 80);
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
  else if(mode == 2){ //towers
    complex<double> z4 = f4(z, time);
    return z4;
  }
  else if(mode == 3){ //beads
    complex<double> z5 = f5(z, time);
    return z5;
  }
  else if(mode == 4){ //planet
    double alpha = M_PI/36.0;
    z = complex<double>(z.real()*cos(alpha) - z.imag()*sin(alpha), z.imag()*cos(alpha) + z.real()*sin(alpha));

    complex<double> z6 = f6(z, time);
    complex<double> z7 = f7(z, time);

    return z6 + z7;
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
