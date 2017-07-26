#ifndef _FFT_H
#define _FFT_H

#include <complex>
#include <cmath>
#include <valarray>

using std::complex;
using namespace std::literals::complex_literals;
typedef std::valarray<complex<double>> carray;
typedef complex<double> cdouble;

enum direction { FORWARD, REVERSE };
const double PI = acos(-1);
void fft(carray& x);
void ifft(carray& x);

#endif
