#ifndef _FT_HELPERS_H
#define _FT_HELPERS_H

#include <complex>
#include <cmath>
#include <valarray>

using std::complex;
using namespace std::literals::complex_literals;
typedef std::valarray<complex<double>> carray;
typedef complex<double> cdouble;

enum direction { FORWARD, REVERSE };
const double PI = acos(-1);

#endif
