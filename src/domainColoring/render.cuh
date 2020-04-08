#ifndef RENDER_H
#define RENDER_H

extern "C" void render(cv::cuda::GpuMat frame, double time, uintmax_t mode);

#endif
