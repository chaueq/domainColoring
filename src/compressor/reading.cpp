#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <sys/resource.h>
#include <unistd.h>
#include "Pixel.hpp"
#include "reading.hpp"

using namespace std;
using namespace cv;

void* readingWorker(void* threadarg)
{
  readingWorkerData* td = (readingWorkerData*) threadarg;

  for(uint64_t i = 0; i < td->share + td->additionalShare; ++i)
  {
    const uint16_t x = (i + (td->share*td->thread_id)) / (td->height);
    const uint16_t y = (i + (td->share*td->thread_id)) % (td->height);
    const RGB value(
      td->frame.data[(3*(i + (td->share*td->thread_id))) + 2],
      td->frame.data[(3*(i + (td->share*td->thread_id))) + 1],
      td->frame.data[(3*(i + (td->share*td->thread_id))) + 0]
    );

    td->data[(x*td->height)+y].consumeValue(value, td->tollerance);
  }

  pthread_exit(NULL);
}
