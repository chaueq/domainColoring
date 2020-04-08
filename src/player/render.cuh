#ifndef RENDER_H
#define RENDER_H

using namespace cv::cuda;

extern "C" void render(GpuMat frame, uint8_t*** values, uint8_t** repetitions, uint8_t** toBeLoaded);

#endif
