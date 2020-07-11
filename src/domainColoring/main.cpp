#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <unistd.h>
#include <sys/resource.h>
#include <chrono>
#include "render.cuh"
#include "cui.hpp"
#include "../libprv/PRVcompressor.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
  if (argc != 4 && argc != 5 && argc != 8)
  {
      cerr << argv[0] << " <mode> <width> <height> [<output_prv_file> <start_time> <duration_time> <fps>]" << endl;
      cerr << endl;
      cerr << "\t" << "width, height - pixels" << endl;
      cerr << "\t" << "start_time, duration_time - in seconds" << endl;
      cerr << endl;
      cerr << "Modes:" << endl;
      cerr << "\t" << "0 - animation" << endl;
      cerr << "\t" << "1 - flower" << endl;
      cerr << "\t" << "2 - towers" << endl;
      cerr << "\t" << "3 - beads" << endl;
      return 1;
  }

  const uintmax_t width = max(atol(argv[2]), atol(argv[3]));
  const uintmax_t height = min(atol(argv[2]), atol(argv[3]));
  const uintmax_t mode = atol(argv[1]);
  cerr << "Resolution: " << width << " × " << height << endl;
  uintmax_t time;

  cuda::GpuMat frameHSL(height, width, CV_8UC3);
  cuda::GpuMat frameBGR(height, width, CV_8UC3);


  if(argc == 4) //DISPLAY MODE
  {
    cerr << "Output: screen" << endl;
    Mat gui(height, width, CV_8UC3), guiResized(1080, 1920, CV_8UC3);

    namedWindow(argv[0], WINDOW_NORMAL);
    setWindowProperty(argv[0], WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    do
    {
      time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
      render(frameHSL, time, mode);
      cuda::cvtColor(frameHSL, frameBGR, COLOR_HLS2BGR);

      frameBGR.download(gui);
      resize(gui, guiResized, Size(1920, 1080));
      imshow(argv[0], guiResized);
    }
    while(waitKey(mode == 0) != 27);

    destroyAllWindows();
  }

  else if(argc == 5)
  {
    const char* path = argv[4];
    cerr << "Ouput: " << path << endl;

    Mat cpuMat;
    time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();

    render(frameHSL, time, mode);
    cuda::cvtColor(frameHSL, frameBGR, COLOR_HLS2BGR);
    frameBGR.download(cpuMat);

    imwrite(path, cpuMat);
  }

  else if(argc == 8)
  {
    const char* path = argv[4];
    const uintmax_t start_time = atol(argv[5]) * 1000;
    const uintmax_t duration_time = atol(argv[6]) * 1000;
    const uintmax_t fps = atol(argv[7]);

    cerr << "Ouput: " << path << endl;
    cerr << "Start time: " << start_time/1000 << endl;
    cerr << "Duration: " << duration_time/1000 << endl;
    cerr << "FPS: " << fps << endl;

    uintmax_t framesDone = 0;
    const uintmax_t framesTBD = (duration_time / 1000) * fps;
    const double time_step = 1000 / (double)fps;
    PRVcompressor output(path, 0, fps, framesTBD, width, height);
    Mat cpuVideoBuffer;

    cerr << "Frames TBG: " << framesTBD << endl;
    cerr << endl;

    pthread_t cuiThread;
    cuiThreadData td;
    void* status;
    td.framesDone = &framesDone;
    td.framesTBD = framesTBD;
    pthread_create(&cuiThread, NULL, cuiWorker, (void*)&td);

    for(; framesDone < framesTBD; ++framesDone)
    {
      time = start_time + (uintmax_t)round((double)framesDone * time_step);
      render(frameHSL, time, mode);
      cuda::cvtColor(frameHSL, frameBGR, COLOR_HLS2BGR);

      frameBGR.download(cpuVideoBuffer);
      output.upload(cpuVideoBuffer);
    }

    pthread_join(cuiThread, &status);
  }

  return 0;
}
