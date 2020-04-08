#include <bits/stdc++.h>
#include "RGB.hpp"

using namespace std;

RGB::RGB(){}

RGB::RGB(uint16_t r, uint16_t g, uint16_t b)
{
  R = r;
  G = g;
  B = b;
}

uint8_t RGB::diffTo(RGB x)
{
  double buffer = sqrt(pow(x.R-R, 2) + pow(x.G-G, 2) + pow(x.B-B, 2));
  uint8_t result = (uint8_t)ceil(buffer * 100 / 442);
  return result;
}
