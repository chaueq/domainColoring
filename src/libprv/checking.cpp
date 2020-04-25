#include <bits/stdc++.h>
#include <sys/resource.h>
#include <unistd.h>
#include "Pixel.hpp"
#include "checking.hpp"

void* checkingWorker(void* threadarg)
{
  checkingWorkerData* td = (checkingWorkerData*) threadarg;

  for(uint64_t i = 0; i < td->share + td->additionalShare; ++i)
  {
    const uint16_t x = (i + (td->share*td->thread_id)) / (td->height);
    const uint16_t y = (i + (td->share*td->thread_id)) % (td->height);

    if(td->repetitions[(x*td->height)+y] == 0){
      if((!td->data[(x*td->height)+y].isDumpable()) && !(td->data[(x*td->height)+y].hasData() && td->readingDone == 1)){
        td->fine = 0;
        break;
      }
    }
  }

  pthread_exit(NULL);
}
