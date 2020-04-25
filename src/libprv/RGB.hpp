#ifndef RGB_HPP
#define RGB_HPP

#include <bits/stdc++.h>

class RGB
{
  public:
    uint16_t R = 0;
    uint16_t G = 0;
    uint16_t B = 0;

    RGB();
    RGB(uint16_t r, uint16_t g, uint16_t b);
    uint8_t diffTo(RGB x);
};

#endif
