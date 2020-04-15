#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <sys/resource.h>
#include <unistd.h>
#include "Pixel.hpp"
#include "writing.hpp"

using namespace std;

void* writingWorker(void* threadarg)
{
  writingWorkerData* td = (writingWorkerData*) threadarg;

  for(uint64_t i = 0; i < td->share + td->additionalShare; ++i)
  {
    const uint16_t x = (i + (td->share*td->thread_id)) / (td->height);
    const uint16_t y = (i + (td->share*td->thread_id)) % (td->height);

    if(td->repetitions[(x*td->height)+y] == 0){
      td->to_dump[td->to_dump_size] = td->data[(x*td->height)+y].dumpValue();
      td->repetitions[(x*td->height)+y] = (td->to_dump[td->to_dump_size] >> 24) % 256;
      td->to_dump_size += 1;
    }

    else
    {
      td->repetitions[(x*td->height)+y] -= 1;
    }
  }

  pthread_exit(NULL);
}
